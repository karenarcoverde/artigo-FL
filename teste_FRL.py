"""
Cell-Free (toy) + FRL-QGradual vs FRL clássico (FedAvg)
=======================================================

Este script implementa:
- APs = agentes (heterogêneos)
- CPU = servidor (agrega modelos)
- FRL clássico (FedAvg): w_t = 1
- FRL-QGradual: w_t começa > 1 e decai até 1

E mede (por rodada) no estilo do artigo:
1) running average win rate
2) AUC do running average win rate
3) rounds-to-target reward

⚠️ Contexto cell-free:
- Aqui o ambiente é "toy" (proxy). No seu simulador cell-free real, você só precisa
  substituir SimpleCellFreeEnv e manter o bloco de métricas do mesmo jeito.
- "Win" é definido via um KPI por rodada. Neste toy, eu deixei por padrão "reward".
  Você pode trocar para "queue" (fila média) se quiser que fique mais 'rede'.

Requisitos:
  pip install numpy torch matplotlib
Rodar:
  python frl_qgradual_metrics_cellfree.py
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# CONFIG (métricas do paper)
# =========================
SEED = 42

# Janela da running average win rate (igual ideia do "running average")
WINRATE_WINDOW = 20

# Janela da média móvel do reward (para rounds-to-target reward)
REWARD_MA_WINDOW = 20

# "Win" por rodada: escolha um KPI e um threshold
# KPI_MODE:
#   - "reward"  -> win se mean_reward_round >= WIN_THRESHOLD
#   - "queue"   -> win se mean_queue_round <= WIN_THRESHOLD (menor é melhor)
KPI_MODE = "reward"
WIN_THRESHOLD = -100.0  # ex.: -100 (para reward negativo). Ajuste conforme seu cenário.

# Target de reward (rounds-to-target reward)
REWARD_TARGET = -100.0       # alvo no reward MA (média móvel)
TARGET_PATIENCE = 10         # precisa manter >= target por N rounds seguidos para contar "convergiu"

# Treino
ROUNDS = 200
N_APS = 8
K_UES = 8
LOCAL_EPISODES_PER_ROUND = 2
HORIZON = 50

# QGradual
DELTA_ROUNDS = 150
CLIP_VMIN = -0.05
CLIP_VMAX = 0.05

DEVICE = "cpu"


# -----------------------------
# Seeds / Reprodutibilidade
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Toy "Cell-Free" Environment
# -----------------------------
class SimpleCellFreeEnv:
    """
    Proxy cell-free:
      - ganhos de canal (K UEs)
      - interferência
      - filas (tráfego)
    Ação: escolhe (UE, potência)
    Reward: rate - alpha*sum(queue) - beta*power
    Retorna info com rate e queue_sum para poder calcular KPI por rodada.
    """

    def __init__(
        self,
        ap_id: int,
        K: int = 8,
        power_levels: Tuple[float, ...] = (0.25, 0.5, 1.0),
        noise: float = 1e-3,
        horizon: int = 50,
    ):
        self.ap_id = ap_id
        self.K = K
        self.power_levels = power_levels
        self.P = len(power_levels)
        self.action_size = K * self.P
        self.noise = noise
        self.horizon = horizon

        # heterogeneidade por AP
        rng = np.random.default_rng(SEED + 1000 + ap_id)
        self.pathloss_scale = float(rng.uniform(0.6, 1.4))
        self.interf_scale = float(rng.uniform(0.5, 2.0))
        self.traffic_scale = float(rng.uniform(0.5, 2.0))

        self.t = 0
        self.gains = np.zeros(self.K, dtype=np.float32)
        self.queues = np.zeros(self.K, dtype=np.float32)
        self.interf = 0.0

    def reset(self) -> np.ndarray:
        self.t = 0
        rng = np.random.default_rng(SEED + 2000 + self.ap_id)
        self.gains = (self.pathloss_scale * rng.lognormal(mean=-0.2, sigma=0.6, size=self.K)).astype(np.float32)
        self.queues = (rng.uniform(0.0, 5.0, size=self.K) * self.traffic_scale).astype(np.float32)
        self.interf = float(self.interf_scale * rng.lognormal(mean=-1.0, sigma=0.4))
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.gains, self.queues, np.array([self.interf], dtype=np.float32)], axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.t += 1
        ue = action // self.P
        p_idx = action % self.P
        power = float(self.power_levels[p_idx])

        rng = np.random.default_rng(SEED + 3000 + self.ap_id + self.t)
        self.gains = (0.8 * self.gains + 0.2 * (self.pathloss_scale * rng.lognormal(-0.2, 0.6, size=self.K))).astype(np.float32)
        self.interf = float(0.7 * self.interf + 0.3 * (self.interf_scale * rng.lognormal(-1.0, 0.4)))

        arrivals = rng.poisson(lam=0.8 * self.traffic_scale, size=self.K).astype(np.float32)
        self.queues += arrivals

        g = float(self.gains[ue])
        sinr = (power * g) / (self.noise + self.interf)
        rate = math.log2(1.0 + sinr)

        served = min(self.queues[ue], rate * 2.0)
        self.queues[ue] -= served

        alpha = 0.02
        beta = 0.05
        reward = float(rate - alpha * float(self.queues.sum()) - beta * power)

        done = self.t >= self.horizon
        info = {
            "ue": ue,
            "power": power,
            "rate": float(rate),
            "queue_sum": float(self.queues.sum()),
            "sinr": float(sinr),
        }
        return self._get_state(), reward, done, info


# -----------------------------
# DQN (rede + replay buffer)
# -----------------------------
class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, tr: Transition) -> None:
        self.buf.append(tr)

    def sample(self, batch_size: int) -> List[Transition]:
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buf)


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)

def unflatten_to_params(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i+n].view_as(p))
        i += n


# -----------------------------
# Servidor FRL (CPU)
# -----------------------------
class FRLServer:
    """
    FedAvg: w_t = 1 sempre
    QGradual: w0=log(n)+1 e w_t decai linearmente até 1 em DELTA_ROUNDS.
    Agregação: Δθ_bar=(w_t/n)*sum(Δθ_k), com clipping.
    """
    def __init__(
        self,
        global_model: nn.Module,
        n_agents: int,
        delta_rounds: int,
        vmin: float,
        vmax: float,
        use_qgradual: bool,
        device: str = "cpu",
    ):
        self.device = device
        self.model = global_model.to(device)
        self.n = int(n_agents)
        self.delta_rounds = max(1, int(delta_rounds))
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.use_qgradual = bool(use_qgradual)
        self.t = 0
        self.w0 = float(math.log(self.n) + 1.0)

    def wt(self) -> float:
        if not self.use_qgradual:
            return 1.0
        if self.t < self.delta_rounds:
            return float(self.w0 - (self.t * (self.w0 - 1.0) / self.delta_rounds))
        return 1.0

    def get_global_flat(self) -> torch.Tensor:
        return flatten_params(self.model).detach().clone()

    def set_global_flat(self, flat: torch.Tensor) -> None:
        unflatten_to_params(self.model, flat.to(self.device))

    def aggregate(self, deltas_flat: List[torch.Tensor]) -> float:
        self.t += 1
        wt = self.wt()
        stacked = torch.stack(deltas_flat, dim=0)  # [n, P]
        delta_bar = (wt / self.n) * stacked.sum(dim=0)
        delta_bar = torch.clamp(delta_bar, self.vmin, self.vmax)
        new_global = self.get_global_flat() + delta_bar.to(self.device)
        self.set_global_flat(new_global)
        return wt


# -----------------------------
# Agente (AP)
# -----------------------------
class APAgent:
    def __init__(self, ap_id: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.ap_id = ap_id
        self.device = device
        self.gamma = 0.99

        self.q = QNet(state_dim, action_dim).to(device)
        self.tgt = QNet(state_dim, action_dim).to(device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=1e-3)
        self.rb = ReplayBuffer()

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.train_steps = 0

    def load_global(self, global_flat: torch.Tensor) -> None:
        unflatten_to_params(self.q, global_flat.to(self.device))
        unflatten_to_params(self.tgt, global_flat.to(self.device))

    @torch.no_grad()
    def act(self, s: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.q.net[-1].out_features)
        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        qv = self.q(x).squeeze(0)
        return int(torch.argmax(qv).item())

    def observe(self, tr: Transition) -> None:
        self.rb.push(tr)

    def local_update(self, batch_size: int = 128, target_sync: int = 200) -> None:
        if len(self.rb) < batch_size:
            return
        batch = self.rb.sample(batch_size)

        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.stack([b.s2 for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_q_next = self.tgt(s2).max(dim=1, keepdim=True).values
            y = r + (1.0 - d) * self.gamma * max_q_next

        loss = (q_sa - y).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def delta_to_global(self, global_flat: torch.Tensor) -> torch.Tensor:
        local_flat = flatten_params(self.q).detach().clone()
        return (local_flat - global_flat.to(self.device)).detach().cpu()


# =========================
# Métricas estilo artigo
# =========================
def running_avg(arr: List[float], window: int) -> float:
    if len(arr) == 0:
        return 0.0
    w = min(window, len(arr))
    return float(np.mean(arr[-w:]))

def auc_trapz(y: List[float]) -> float:
    if len(y) <= 1:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.trapz(np.array(y, dtype=float), x))

def is_win(mean_reward_round: float, mean_queue_round: float) -> int:
    if KPI_MODE == "reward":
        return int(mean_reward_round >= WIN_THRESHOLD)
    if KPI_MODE == "queue":
        return int(mean_queue_round <= WIN_THRESHOLD)
    raise ValueError("KPI_MODE inválido. Use 'reward' ou 'queue'.")


def train_and_measure(method_name: str, use_qgradual: bool) -> Dict:
    # envs e dimensões
    envs = [SimpleCellFreeEnv(ap_id=i, K=K_UES, horizon=HORIZON) for i in range(N_APS)]
    state_dim = K_UES + K_UES + 1
    action_dim = envs[0].action_size

    # servidor + agentes
    global_model = QNet(state_dim, action_dim)
    server = FRLServer(
        global_model=global_model,
        n_agents=N_APS,
        delta_rounds=DELTA_ROUNDS,
        vmin=CLIP_VMIN,
        vmax=CLIP_VMAX,
        use_qgradual=use_qgradual,
        device=DEVICE,
    )

    agents = [APAgent(ap_id=i, state_dim=state_dim, action_dim=action_dim, device=DEVICE) for i in range(N_APS)]
    g0 = server.get_global_flat()
    for ag in agents:
        ag.load_global(g0)

    # fronthaul (uplink)
    num_params = int(server.get_global_flat().numel())
    bytes_per_ap_per_round = num_params * 4

    uplink_bytes_total = 0

    # séries por rodada
    mean_reward_series: List[float] = []
    mean_queue_series: List[float] = []
    win_series: List[int] = []
    running_winrate_series: List[float] = []
    reward_ma_series: List[float] = []

    # rounds-to-target reward (com patience)
    hit_streak = 0
    stop_round_reward_target: Optional[int] = None

    print(f"\n=== {method_name} ===")
    for rnd in range(1, ROUNDS + 1):
        global_flat = server.get_global_flat()
        deltas: List[torch.Tensor] = []

        # agregados por round (para KPI)
        round_rewards_per_ap = []
        round_queue_means_per_ap = []

        # treino local em cada AP
        for env, ag in zip(envs, agents):
            ag.load_global(global_flat)

            total_r = 0.0
            queue_sum_acc = 0.0
            rate_acc = 0.0
            steps_acc = 0

            for _ in range(LOCAL_EPISODES_PER_ROUND):
                s = env.reset()
                done = False
                while not done:
                    a = ag.act(s)
                    s2, r, done, info = env.step(a)

                    ag.observe(Transition(s=s, a=a, r=r, s2=s2, done=done))
                    ag.local_update(batch_size=128, target_sync=200)

                    total_r += r
                    queue_sum_acc += float(info["queue_sum"])
                    rate_acc += float(info["rate"])
                    steps_acc += 1
                    s = s2

            # reward médio por episódio (igual ao seu código original)
            r_round_ap = total_r / max(1, LOCAL_EPISODES_PER_ROUND)
            round_rewards_per_ap.append(r_round_ap)

            # KPI de rede (proxy): média da fila total ao longo dos steps locais
            q_mean_ap = queue_sum_acc / max(1, steps_acc)
            round_queue_means_per_ap.append(q_mean_ap)

            # uplink (SEM compressão): cada AP envia Δθ completo float32
            delta = ag.delta_to_global(global_flat)
            deltas.append(delta)
            uplink_bytes_total += bytes_per_ap_per_round

        # agregação no servidor
        wt = server.aggregate(deltas)

        # KPI global por round (média entre APs)
        mean_reward_round = float(np.mean(round_rewards_per_ap))
        mean_queue_round = float(np.mean(round_queue_means_per_ap))

        mean_reward_series.append(mean_reward_round)
        mean_queue_series.append(mean_queue_round)

        # WIN e running average win rate
        w = is_win(mean_reward_round, mean_queue_round)
        win_series.append(w)
        run_winrate = running_avg(win_series, WINRATE_WINDOW)
        running_winrate_series.append(run_winrate)

        # reward MA (para rounds-to-target reward)
        reward_ma = running_avg(mean_reward_series, REWARD_MA_WINDOW)
        reward_ma_series.append(reward_ma)

        # rounds-to-target reward (com patience)
        if reward_ma >= REWARD_TARGET:
            hit_streak += 1
        else:
            hit_streak = 0
        if stop_round_reward_target is None and hit_streak >= TARGET_PATIENCE:
            stop_round_reward_target = rnd

        # log por rodada
        print(
            f"[Round {rnd:04d}] "
            f"w_t={wt:.3f} | "
            f"mean_reward={mean_reward_round:.3f} | reward_MA={reward_ma:.3f} | "
            f"mean_queue={mean_queue_round:.3f} | "
            f"win={w} | run_winrate={run_winrate:.3f} | "
            f"uplink_total={uplink_bytes_total/1024/1024:.2f} MB"
        )

    # AUC da running average win rate
    auc_winrate = auc_trapz(running_winrate_series)

    return {
        "method": method_name,
        "use_qgradual": use_qgradual,
        "num_params": num_params,
        "bytes_per_ap_per_round": bytes_per_ap_per_round,
        "uplink_total_bytes": uplink_bytes_total,
        "mean_reward_series": mean_reward_series,
        "mean_queue_series": mean_queue_series,
        "win_series": win_series,
        "running_winrate_series": running_winrate_series,
        "reward_ma_series": reward_ma_series,
        "auc_running_winrate": auc_winrate,
        "rounds_to_target_reward": stop_round_reward_target,
    }


def print_summary(res_a: Dict, res_b: Dict) -> None:
    print("\n==================== SUMMARY ====================")
    print(f"KPI_MODE = {KPI_MODE} | WIN_THRESHOLD = {WIN_THRESHOLD}")
    print(f"REWARD_TARGET = {REWARD_TARGET} | PATIENCE = {TARGET_PATIENCE}")
    print("------------------------------------------------")

    for r in (res_a, res_b):
        rt = r["rounds_to_target_reward"]
        rt_str = str(rt) if rt is not None else "None (não atingiu)"
        print(
            f"{r['method']}\n"
            f"  AUC(run_winrate)      : {r['auc_running_winrate']:.4f}\n"
            f"  rounds-to-target(rew) : {rt_str}\n"
            f"  uplink total          : {r['uplink_total_bytes']/1024/1024:.2f} MB\n"
        )

    # Comparação direta
    rt_a = res_a["rounds_to_target_reward"]
    rt_b = res_b["rounds_to_target_reward"]
    if rt_a is not None and rt_b is not None:
        print(f"Δ rounds-to-target = {rt_a - rt_b:+d} (positivo => {res_b['method']} atingiu antes)")
    print("================================================\n")


def plot_results(res_a: Dict, res_b: Dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib não disponível:", e)
        return

    x = np.arange(1, ROUNDS + 1)

    plt.figure()
    plt.plot(x, res_a["running_winrate_series"], label=f"{res_a['method']} run_winrate")
    plt.plot(x, res_b["running_winrate_series"], label=f"{res_b['method']} run_winrate")
    plt.xlabel("Round")
    plt.ylabel("Running average win rate")
    plt.legend()
    plt.title("Running average win rate (estilo paper)")
    plt.show()

    plt.figure()
    plt.plot(x, res_a["reward_ma_series"], label=f"{res_a['method']} reward_MA")
    plt.plot(x, res_b["reward_ma_series"], label=f"{res_b['method']} reward_MA")
    plt.axhline(REWARD_TARGET, linestyle="--", label="reward target")
    plt.xlabel("Round")
    plt.ylabel("Reward moving average")
    plt.legend()
    plt.title("Rounds-to-target reward (cruzamento do alvo)")
    plt.show()


if __name__ == "__main__":
    # FRL clássico = FedAvg (w=1)
    res_frl = train_and_measure("FRL clássico (FedAvg)", use_qgradual=False)

    # FRL-QGradual
    res_qg = train_and_measure("FRL-QGradual", use_qgradual=True)

    print_summary(res_frl, res_qg)
    plot_results(res_frl, res_qg)
