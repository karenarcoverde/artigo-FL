# frl_qgradual_cellfree.py
# ============================================================
# FRL-QGradual (DQN-FedGradual) no contexto Cell-Free (toy env)
# APs = agentes, CPU = servidor federado
#
# Modos:
#   --algo qgradual : FRL-QGradual (DQN local + w_t gradual)
#   --algo fedavg   : FRL-FedAvg   (DQN local + w_t=1)
#   --algo fl       : FL puro (supervisionado + FedAvg), sem RL/TD
#
# Comunicação (uplink AP->CPU):
#   --comm_mode none        : envia delta full float32 (baseline)
#   --comm_mode topk_quant  : envia delta esparso top-k + quantização (bits menores)
#   --upload_thresh > 0     : se ||delta||_2 < thresh -> não envia nada (0 bits)
#
# Métricas principais:
#   - rounds_to_target (via reward MA)
#   - total_uplink_bits_to_target (bits até convergir)
#   - total_uplink_bits (bits no horizonte total)
#
# Dependências: numpy, torch
# ============================================================

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# -------------------------
# Reprodutibilidade
# -------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Replay Buffer (simples)
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim

        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.r = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.s2 = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.d = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, s, a, r, s2, done):
        idx = self.ptr
        self.s[idx] = torch.tensor(s, dtype=torch.float32, device=self.device)
        self.a[idx] = torch.tensor([a], dtype=torch.int64, device=self.device)
        self.r[idx] = torch.tensor([r], dtype=torch.float32, device=self.device)
        self.s2[idx] = torch.tensor(s2, dtype=torch.float32, device=self.device)
        self.d[idx] = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


# -------------------------
# Q-Network
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Teacher para FL puro
# -------------------------
def teacher_action_beta_queue(
    obs: np.ndarray, K: int, n_powers: int, power_idx_fixed: int = 1
) -> int:
    beta = obs[:K]
    q = obs[K:2 * K] + 1e-3
    score = beta / q
    ue = int(np.argmax(score))
    return ue * n_powers + int(power_idx_fixed)


# -------------------------
# Ambiente Cell-Free (toy)
# -------------------------
@dataclass
class CellFreeConfig:
    L: int = 5
    K: int = 10
    power_levels: Tuple[float, ...] = (0.2, 0.6, 1.0)
    noise: float = 1e-2
    pathloss_exp: float = 3.5
    area_km: float = 0.5
    episode_len: int = 40
    arrival_rate: float = 0.6
    queue_max: int = 20
    reward_lambda_comm: float = 0.02
    reward_lambda_queue: float = 0.01


class CellFreeToyEnv:
    def __init__(self, cfg: CellFreeConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.ap_pos = self.rng.uniform(0, cfg.area_km, size=(cfg.L, 2))
        self.ue_pos = self.rng.uniform(0, cfg.area_km, size=(cfg.K, 2))

        self.beta = self._compute_pathloss(self.ap_pos, self.ue_pos, cfg.pathloss_exp)

        self.t = 0
        self.queues = np.zeros(cfg.K, dtype=np.float32)
        self.h = None

        self.n_actions = cfg.K * len(cfg.power_levels)
        self.obs_dim = cfg.K + cfg.K + 1

    def _compute_pathloss(self, ap_pos, ue_pos, expn):
        d = np.linalg.norm(ap_pos[:, None, :] - ue_pos[None, :, :], axis=2) + 1e-3
        beta = 1.0 / (d ** expn)
        beta = beta / (beta.max() + 1e-12)
        return beta.astype(np.float32)

    def reset(self):
        self.t = 0
        self.queues[:] = self.rng.integers(0, 5, size=self.cfg.K).astype(np.float32)
        self._sample_channels()
        return self._get_obs_all()

    def _sample_channels(self):
        g2 = self.rng.exponential(scale=1.0, size=(self.cfg.L, self.cfg.K)).astype(np.float32)
        self.h = self.beta * g2

    def _decode_action(self, a: int):
        p_levels = self.cfg.power_levels
        ue = a // len(p_levels)
        p_idx = a % len(p_levels)
        p = p_levels[p_idx]
        return ue, p

    def _get_obs_one(self, l: int):
        step_norm = np.array([self.t / max(1, self.cfg.episode_len - 1)], dtype=np.float32)
        obs = np.concatenate([self.beta[l], self.queues / self.cfg.queue_max, step_norm], axis=0)
        return obs

    def _get_obs_all(self):
        return [self._get_obs_one(l) for l in range(self.cfg.L)]

    def step(self, actions: List[int]):
        cfg = self.cfg
        L, K = cfg.L, cfg.K

        arrivals = self.rng.poisson(lam=cfg.arrival_rate, size=K).astype(np.float32)
        self.queues = np.clip(self.queues + arrivals, 0, cfg.queue_max)

        chosen_ue = np.zeros(L, dtype=np.int64)
        chosen_p = np.zeros(L, dtype=np.float32)
        for l, a in enumerate(actions):
            ue, p = self._decode_action(int(a))
            chosen_ue[l] = ue
            chosen_p[l] = p

        signal = np.zeros(K, dtype=np.float32)
        interf = np.zeros(K, dtype=np.float32)
        for l in range(L):
            k = chosen_ue[l]
            signal[k] += chosen_p[l] * self.h[l, k]
            for kk in range(K):
                if kk != k:
                    interf[kk] += chosen_p[l] * self.h[l, kk]

        sinr = signal / (interf + cfg.noise)
        rate = np.log2(1.0 + sinr)

        served = np.minimum(self.queues, rate * 2.0)
        self.queues = np.clip(self.queues - served, 0, cfg.queue_max)

        rewards = np.zeros(L, dtype=np.float32)
        backlog_pen = cfg.reward_lambda_queue * float(self.queues.sum())
        for l in range(L):
            k = chosen_ue[l]
            power_pen = cfg.reward_lambda_comm * chosen_p[l]
            rewards[l] = float(rate[k]) - power_pen - backlog_pen

        self.t += 1
        done = (self.t >= cfg.episode_len)
        self._sample_channels()
        obs_next = self._get_obs_all()

        info = {"sum_rate": float(rate.sum()), "avg_queue": float(self.queues.mean())}
        return obs_next, rewards.tolist(), done, info


# -------------------------
# DQN Agent (FRL)
# -------------------------
@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50000
    start_learn: int = 200          # <<-- mais rápido que 1000
    train_freq: int = 1
    target_update: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 8000     # <<-- mais rápido que 20000
    grad_clip_norm: float = 10.0


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: torch.device, seed: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions).to(device)
        self.q_tgt = QNetwork(obs_dim, n_actions).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.buffer_size, obs_dim, device)
        self.total_steps = 0

    def epsilon(self):
        cfg = self.cfg
        t = min(self.total_steps, cfg.eps_decay_steps)
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (t / cfg.eps_decay_steps)
        return float(eps)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        eps = self.epsilon()
        self.total_steps += 1
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        qvals = self.q(x)
        return int(torch.argmax(qvals, dim=1).item())

    def store(self, s, a, r, s2, done):
        self.rb.add(s, a, r, s2, done)

    def maybe_train(self):
        cfg = self.cfg
        if self.rb.size < cfg.start_learn:
            return 0.0
        if (self.total_steps % cfg.train_freq) != 0:
            return 0.0

        s, a, r, s2, d = self.rb.sample(cfg.batch_size)

        with torch.no_grad():
            next_a = torch.argmax(self.q(s2), dim=1, keepdim=True)
            next_q = self.q_tgt(s2).gather(1, next_a)
            y = r + cfg.gamma * (1.0 - d) * next_q

        qsa = self.q(s).gather(1, a)
        loss = nn.functional.smooth_l1_loss(qsa, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip_norm)
        self.opt.step()

        if (self.total_steps % cfg.target_update) == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def get_param_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.q.parameters()).detach().clone()

    def set_param_vector(self, vec: torch.Tensor):
        vector_to_parameters(vec.detach().clone(), self.q.parameters())
        self.q_tgt.load_state_dict(self.q.state_dict())


# -------------------------
# FL Agent (supervisionado)
# -------------------------
class FLAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float, device: torch.device, seed: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.model = QNetwork(obs_dim, n_actions).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.model(x)
        return int(torch.argmax(logits, dim=1).item())

    def train_supervised_batch(self, obs_batch: np.ndarray, a_batch: np.ndarray) -> float:
        x = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        y = torch.tensor(a_batch, dtype=torch.int64, device=self.device)

        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def get_param_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def set_param_vector(self, vec: torch.Tensor):
        vector_to_parameters(vec.detach().clone(), self.model.parameters())


# -------------------------
# Servidor Federado (CPU)
# -------------------------
@dataclass
class FedConfig:
    rounds: int = 200
    local_episodes_per_round: int = 2     # <<-- mais local update por round (acelera FRL)
    delta_rounds: int = 60
    warmup_rounds: int = 30              # <<-- impede wt>1 no início (estabiliza)
    vmin: float = -100.0
    vmax: float = 100.0


class FedServer:
    def __init__(self, global_init: torch.Tensor, n_agents: int, fedcfg: FedConfig, algo: str):
        self.theta = global_init.detach().clone()
        self.n = n_agents
        self.fedcfg = fedcfg
        assert algo in ("qgradual", "fedavg")
        self.algo = algo
        self.w0 = float(math.log(n_agents) + 1.0)

    def weight_wt(self, t_round: int) -> float:
        # warmup: durante as primeiras rodadas, wt=1 pra não amplificar ruído
        if t_round < int(self.fedcfg.warmup_rounds):
            return 1.0
        if self.algo == "fedavg":
            return 1.0

        delta = max(1, int(self.fedcfg.delta_rounds))
        # t_round aqui é absoluto; vamos mapear para "tempo após warmup"
        tw = t_round - int(self.fedcfg.warmup_rounds)
        if 0 < tw < delta:
            return self.w0 - (tw * (self.w0 - 1.0) / delta)
        return 1.0

    def aggregate(self, deltas: List[torch.Tensor], t_round: int) -> torch.Tensor:
        wt = self.weight_wt(t_round)
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        agg = wt * mean_delta
        agg = torch.clamp(agg, self.fedcfg.vmin, self.fedcfg.vmax)
        self.theta = self.theta + agg
        return agg


# -------------------------
# Comunicação: top-k + quantização + gatilho
# -------------------------
def compress_delta_topk_quant(
    delta: torch.Tensor,
    topk_frac: float,
    quant_bits: int,
    upload_thresh: float,
) -> Tuple[torch.Tensor, int]:
    """
    Retorna:
      delta_hat (reconstruído no servidor) e bits_uplink (aproximado)
    """
    assert 1 <= quant_bits <= 16
    N = delta.numel()

    # gatilho: não envia nada se delta pequeno
    if upload_thresh > 0.0:
        if float(torch.norm(delta).item()) < upload_thresh:
            return torch.zeros_like(delta), 0

    # top-k
    k = max(1, int(math.ceil(topk_frac * N)))
    vals, idx = torch.topk(delta.abs(), k, largest=True, sorted=False)
    top_idx = idx
    top_val = delta[top_idx]

    # quantização uniforme simétrica nos top-k
    # usa scale = max(|v|)
    vmax = float(top_val.abs().max().item()) + 1e-12
    qmax = (2 ** (quant_bits - 1)) - 1  # signed
    scale = vmax / qmax

    q = torch.clamp(torch.round(top_val / scale), -qmax, qmax).to(torch.int16)
    deq = (q.to(delta.dtype) * scale)

    # reconstrói delta_hat esparso
    delta_hat = torch.zeros_like(delta)
    delta_hat[top_idx] = deq

    # bits: índices + valores + scale
    # - índice: ceil(log2(N)) bits cada
    # - valor quant: quant_bits bits cada
    # - scale: 32 bits (float32) por update
    bits_idx = int(math.ceil(math.log2(N))) if N > 1 else 1
    bits = k * (bits_idx + quant_bits) + 32

    return delta_hat, int(bits)


# -------------------------
# Métricas
# -------------------------
def moving_average(x: List[float], w: int) -> List[float]:
    if w <= 1:
        return x[:]
    out = []
    s = 0.0
    q = []
    for v in x:
        q.append(v)
        s += v
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def auc(y: List[float]) -> float:
    return float(np.sum(np.array(y, dtype=np.float64)))


# -------------------------
# Loop principal
# -------------------------
def run_experiment(
    algo: str,
    seed: int,
    env_cfg: CellFreeConfig,
    dqn_cfg: DQNConfig,
    fed_cfg: FedConfig,
    device: torch.device,
    target_reward: float,
    ma_window: int,
    comm_mode: str,
    topk_frac: float,
    quant_bits: int,
    upload_thresh: float,
):
    set_seeds(seed)
    env = CellFreeToyEnv(env_cfg, seed=seed)

    # cria agentes
    agents = []
    if algo == "fl":
        for l in range(env_cfg.L):
            agents.append(FLAgent(env.obs_dim, env.n_actions, lr=dqn_cfg.lr, device=device, seed=seed + 1000 + l))
    else:
        for l in range(env_cfg.L):
            agents.append(DQNAgent(env.obs_dim, env.n_actions, dqn_cfg, device, seed=seed + 1000 + l))

    # servidor
    theta0 = agents[0].get_param_vector()
    server_algo = "fedavg" if algo == "fl" else algo
    server = FedServer(theta0, n_agents=env_cfg.L, fedcfg=fed_cfg, algo=server_algo)

    for ag in agents:
        ag.set_param_vector(server.theta)

    # métricas
    round_rewards_mean: List[float] = []
    round_wt: List[float] = []
    total_uplink_bits = 0
    total_uplink_bits_to_target: Optional[int] = None
    rounds_to_target: Optional[int] = None

    for t_round in range(fed_cfg.rounds):
        # broadcast
        for ag in agents:
            ag.set_param_vector(server.theta)

        obs_all = env.reset()
        local_round_rewards = []

        # local episodes
        for _ in range(fed_cfg.local_episodes_per_round):
            done = False
            while not done:
                actions = [agents[l].act(obs_all[l]) for l in range(env_cfg.L)]
                obs2_all, rewards_all, done, _info = env.step(actions)

                if algo == "fl":
                    # treino supervisionado
                    n_powers = len(env_cfg.power_levels)
                    K = env_cfg.K
                    labels = [
                        teacher_action_beta_queue(obs_all[l], K=K, n_powers=n_powers, power_idx_fixed=1)
                        for l in range(env_cfg.L)
                    ]
                    for l in range(env_cfg.L):
                        agents[l].train_supervised_batch(
                            obs_batch=np.asarray([obs_all[l]], dtype=np.float32),
                            a_batch=np.asarray([labels[l]], dtype=np.int64),
                        )
                else:
                    # treino DQN
                    for l in range(env_cfg.L):
                        agents[l].store(obs_all[l], actions[l], rewards_all[l], obs2_all[l], done)
                        agents[l].maybe_train()

                obs_all = obs2_all
                local_round_rewards.append(float(np.mean(rewards_all)))

        # deltas + comunicação
        deltas_to_server = []
        bits_this_round = 0

        for ag in agents:
            theta_k = ag.get_param_vector()
            delta = theta_k - server.theta

            if algo == "fl":
                # baseline FL: envia full float32
                # bits = N * 32
                bits_this_round += int(delta.numel()) * 32
                deltas_to_server.append(delta)
            else:
                if comm_mode == "none":
                    bits_this_round += int(delta.numel()) * 32
                    deltas_to_server.append(delta)
                elif comm_mode == "topk_quant":
                    delta_hat, bits = compress_delta_topk_quant(
                        delta=delta,
                        topk_frac=topk_frac,
                        quant_bits=quant_bits,
                        upload_thresh=upload_thresh,
                    )
                    bits_this_round += bits
                    deltas_to_server.append(delta_hat)
                else:
                    raise ValueError("comm_mode inválido")

        total_uplink_bits += bits_this_round

        # agrega
        wt = server.weight_wt(t_round)
        server.aggregate(deltas_to_server, t_round=t_round)
        round_wt.append(float(wt))

        # reward
        rr = float(np.mean(local_round_rewards)) if local_round_rewards else 0.0
        round_rewards_mean.append(rr)

        # convergência
        r_ma = moving_average(round_rewards_mean, ma_window)[-1]
        if rounds_to_target is None and r_ma >= target_reward:
            rounds_to_target = t_round + 1
            total_uplink_bits_to_target = total_uplink_bits

        if (t_round + 1) % 20 == 0:
            print(
                f"[{algo}] round {t_round+1:04d}/{fed_cfg.rounds} "
                f"wt={wt:.3f} reward_mean={rr:.3f} reward_MA({ma_window})={r_ma:.3f} "
                f"uplink_this_round={bits_this_round/1e6:.3f} Mbits "
                f"uplink_total={total_uplink_bits/1e6:.1f} Mbits"
            )

    rewards_ma = moving_average(round_rewards_mean, ma_window)
    return {
        "algo_client": algo,
        "algo_server": server_algo,
        "comm_mode": comm_mode,
        "topk_frac": topk_frac,
        "quant_bits": quant_bits,
        "upload_thresh": upload_thresh,
        "rounds": fed_cfg.rounds,
        "uplink_total_bits": int(total_uplink_bits),
        "uplink_total_bits_to_target": None if total_uplink_bits_to_target is None else int(total_uplink_bits_to_target),
        "rounds_to_target": rounds_to_target,
        "auc_rewards_ma": auc(rewards_ma),
        "last_reward_ma": float(rewards_ma[-1]) if rewards_ma else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="qgradual", choices=["qgradual", "fedavg", "fl"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--delta_rounds", type=int, default=60)
    ap.add_argument("--warmup_rounds", type=int, default=30)
    ap.add_argument("--local_episodes_per_round", type=int, default=2)

    ap.add_argument("--episode_len", type=int, default=40)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--K", type=int, default=10)

    ap.add_argument("--target_reward", type=float, default=4.0)
    ap.add_argument("--ma_window", type=int, default=10)

    # comunicação
    ap.add_argument("--comm_mode", type=str, default="topk_quant", choices=["none", "topk_quant"])
    ap.add_argument("--topk_frac", type=float, default=0.02)      # 2% do vetor
    ap.add_argument("--quant_bits", type=int, default=8)          # 8-bit
    ap.add_argument("--upload_thresh", type=float, default=0.0)   # 0 => sempre envia

    # DQN knobs (pra você ajustar sem mexer no código)
    ap.add_argument("--start_learn", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eps_decay_steps", type=int, default=8000)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    env_cfg = CellFreeConfig(L=args.L, K=args.K, episode_len=args.episode_len)

    dqn_cfg = DQNConfig(
        start_learn=args.start_learn,
        batch_size=args.batch_size,
        eps_decay_steps=args.eps_decay_steps,
    )

    fed_cfg = FedConfig(
        rounds=args.rounds,
        delta_rounds=args.delta_rounds,
        warmup_rounds=args.warmup_rounds,
        local_episodes_per_round=args.local_episodes_per_round,
    )

    res = run_experiment(
        algo=args.algo,
        seed=args.seed,
        env_cfg=env_cfg,
        dqn_cfg=dqn_cfg,
        fed_cfg=fed_cfg,
        device=device,
        target_reward=args.target_reward,
        ma_window=args.ma_window,
        comm_mode=args.comm_mode,
        topk_frac=args.topk_frac,
        quant_bits=args.quant_bits,
        upload_thresh=args.upload_thresh,
    )

    print("\n==================== RESULTADOS ====================")
    print(f"Algo (cliente): {res['algo_client']}")
    print(f"Algo (servidor): {res['algo_server']}")
    print(f"Comm: {res['comm_mode']} | topk={res['topk_frac']*100:.2f}% | q={res['quant_bits']} bits | thresh={res['upload_thresh']}")
    print(f"Rounds: {res['rounds']}")
    print(f"Rounds-to-target (MA >= {args.target_reward}): {res['rounds_to_target']}")
    print(f"Uplink total: {res['uplink_total_bits']/1e6:.2f} Mbits")
    if res["uplink_total_bits_to_target"] is None:
        print("Uplink até target: None (não convergiu)")
    else:
        print(f"Uplink até target: {res['uplink_total_bits_to_target']/1e6:.2f} Mbits")
    print(f"AUC(reward_ma): {res['auc_rewards_ma']:.3f}")
    print(f"Last reward_MA: {res['last_reward_ma']:.3f}")

    print("\nSugestão de comparação (mesmos parâmetros):")
    print("  # baseline FL (sem compressão):")
    print("  python frl_qgradual_cellfree.py --algo fl --comm_mode none")
    print("  # FRL-QGradual com compressão e (opcional) gatilho:")
    print("  python frl_qgradual_cellfree.py --algo qgradual --comm_mode topk_quant --topk_frac 0.02 --quant_bits 8 --upload_thresh 0.0")


if __name__ == "__main__":
    main()
