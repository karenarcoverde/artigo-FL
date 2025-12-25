# frl_qgradual_cellfree.py
# ============================================================
# Comparação (toy Cell-Free):
#   (A) FRL-QGradual / FRL-FedAvg:
#       - Cada AP treina um DQN localmente
#       - Uplink AP->CPU: DELTA de parâmetros (float32) SOMENTE a cada comm_period rounds
#         (sem top-k / sem quantização)
#
#   (B) "FL clássico com dados brutos" (fl_raw):
#       - APs NÃO treinam (só executam a política global + epsilon-greedy)
#       - Uplink AP->CPU: TRANSIÇÕES brutas (s,a,r,s2,done)
#       - CPU treina um DQN centralizado
#
# Métricas:
#   - rounds_to_target (reward moving-average >= target_reward)
#   - uplink_total_bits_to_target
#   - uplink_total_bits
#   - env_steps_to_target, env_steps_total
#   - auc(reward_ma), last_reward_ma
#
# Dependências: numpy, torch
# ============================================================

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
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
# Config DQN
# -------------------------
@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50000
    start_learn: int = 200
    train_freq: int = 1
    target_update: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 8000
    grad_clip_norm: float = 10.0


# -------------------------
# FRL: agente DQN local (treina local)
# -------------------------
class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int,
        cfg: DQNConfig,
        device: torch.device,
        seed: int,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, hidden=hidden).to(device)
        self.q_tgt = QNetwork(obs_dim, n_actions, hidden=hidden).to(device)
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
# "FL clássico com dados brutos": AP é coletor (não treina)
# -------------------------
class CollectorAgent:
    """
    Coleta transições e faz uplink de dados brutos (s,a,r,s2,done).
    Não treina localmente; só executa a política global (broadcast) + epsilon-greedy.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int,
        cfg: DQNConfig,
        device: torch.device,
        seed: int,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, hidden=hidden).to(device)
        self.total_steps = 0
        self.transitions = []  # lista de (s,a,r,s2,done)

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

    def store_transition(self, s, a, r, s2, done):
        self.transitions.append((s, a, r, s2, done))

    def pop_all_transitions(self):
        out = self.transitions
        self.transitions = []
        return out

    def set_param_vector(self, vec: torch.Tensor):
        vector_to_parameters(vec.detach().clone(), self.q.parameters())


# -------------------------
# Servidor federado (FRL): agrega deltas com wt (QGradual) ou wt=1 (FedAvg)
# -------------------------
@dataclass
class FedConfig:
    rounds: int = 200
    local_episodes_per_round: int = 2
    delta_rounds: int = 60
    warmup_rounds: int = 30
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
        # warmup: durante as primeiras rodadas, wt=1 (estabiliza)
        if t_round < int(self.fedcfg.warmup_rounds):
            return 1.0
        if self.algo == "fedavg":
            return 1.0

        delta = max(1, int(self.fedcfg.delta_rounds))
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
# Servidor central (fl_raw): treina DQN global com transições brutas recebidas
# -------------------------
class CentralDQNServer:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int,
        cfg: DQNConfig,
        device: torch.device,
        seed: int,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, hidden=hidden).to(device)
        self.q_tgt = QNetwork(obs_dim, n_actions, hidden=hidden).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.buffer_size, obs_dim, device)
        self.total_steps = 0

    def get_param_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.q.parameters()).detach().clone()

    def ingest_transitions(self, transitions: List[tuple]):
        for (s, a, r, s2, done) in transitions:
            self.rb.add(s, a, r, s2, done)

    def train_steps(self, n_steps: int) -> float:
        cfg = self.cfg
        if self.rb.size < cfg.start_learn:
            return 0.0

        losses = []
        for _ in range(n_steps):
            self.total_steps += 1
            if (self.total_steps % cfg.train_freq) != 0:
                continue

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
            losses.append(float(loss.item()))

            if (self.total_steps % cfg.target_update) == 0:
                self.q_tgt.load_state_dict(self.q.state_dict())

        return float(np.mean(losses)) if losses else 0.0


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
# Contagem de bits (uplink AP->CPU)
# -------------------------
def bits_delta_params_float32(n_params: int) -> int:
    return int(n_params) * 32


def bits_raw_transition(obs_dim: int) -> int:
    """
    Transição bruta: (s, a, r, s2, done)
      s   : obs_dim float32
      s2  : obs_dim float32
      a   : int32  (32 bits)
      r   : float32
      done: int32  (32 bits)
    Total bits = (2*obs_dim + 3) * 32
    """
    return (2 * int(obs_dim) + 3) * 32


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
    hidden: int,
    comm_period: int,
    target_reward: float,
    ma_window: int,
    server_train_steps_per_round: int,
    log_every: int = 20,
) -> Dict:
    """
    algo:
      - "qgradual": FRL-QGradual (uplink = delta parâmetros, comm_period)
      - "fedavg"  : FRL-FedAvg   (uplink = delta parâmetros, comm_period)
      - "fl_raw"  : "FL clássico" com dados brutos (uplink = transições)
    """
    set_seeds(seed)
    env = CellFreeToyEnv(env_cfg, seed=seed)

    round_rewards_mean: List[float] = []
    total_uplink_bits = 0
    total_uplink_bits_to_target: Optional[int] = None
    rounds_to_target: Optional[int] = None

    env_steps_total = 0
    env_steps_to_target: Optional[int] = None

    if algo in ("qgradual", "fedavg"):
        # ---------------- FRL ----------------
        agents = [
            DQNAgent(env.obs_dim, env.n_actions, hidden, dqn_cfg, device, seed=seed + 1000 + l)
            for l in range(env_cfg.L)
        ]
        theta0 = agents[0].get_param_vector()
        server = FedServer(theta0, n_agents=env_cfg.L, fedcfg=fed_cfg, algo=algo)

        for ag in agents:
            ag.set_param_vector(server.theta)

        n_params = int(server.theta.numel())
        comm_period = max(1, int(comm_period))

        for t_round in range(fed_cfg.rounds):
            # broadcast CPU->AP (não contamos)
            for ag in agents:
                ag.set_param_vector(server.theta)

            obs_all = env.reset()
            local_round_rewards = []

            # local episodes: AP treina DQN local
            for _ in range(fed_cfg.local_episodes_per_round):
                done = False
                while not done:
                    actions = [agents[l].act(obs_all[l]) for l in range(env_cfg.L)]
                    obs2_all, rewards_all, done, _info = env.step(actions)

                    for l in range(env_cfg.L):
                        agents[l].store(obs_all[l], actions[l], rewards_all[l], obs2_all[l], done)
                        agents[l].maybe_train()

                    obs_all = obs2_all
                    local_round_rewards.append(float(np.mean(rewards_all)))

                    # conta env steps (um step do ambiente por tempo, independente de L)
                    env_steps_total += 1

            # uplink: deltas SOMENTE quando do_comm=True
            do_comm = ((t_round % comm_period) == 0)
            deltas_to_server = []
            bits_this_round = 0

            if do_comm:
                for ag in agents:
                    theta_k = ag.get_param_vector()
                    delta = theta_k - server.theta
                    deltas_to_server.append(delta)
                    bits_this_round += bits_delta_params_float32(n_params)

                total_uplink_bits += bits_this_round
                server.aggregate(deltas_to_server, t_round=t_round)
            else:
                # sem comunicação: nada enviado, modelo global não muda
                bits_this_round = 0

            rr = float(np.mean(local_round_rewards)) if local_round_rewards else 0.0
            round_rewards_mean.append(rr)

            rewards_ma = moving_average(round_rewards_mean, ma_window)
            r_ma = rewards_ma[-1] if rewards_ma else 0.0

            if rounds_to_target is None and r_ma >= target_reward:
                rounds_to_target = t_round + 1
                total_uplink_bits_to_target = total_uplink_bits
                env_steps_to_target = env_steps_total

            if log_every > 0 and ((t_round + 1) % log_every == 0):
                wt = server.weight_wt(t_round)
                print(
                    f"[{algo}] round {t_round+1:04d}/{fed_cfg.rounds} "
                    f"comm={'Y' if do_comm else 'N'} wt={wt:.3f} "
                    f"reward_mean={rr:.3f} reward_MA({ma_window})={r_ma:.3f} "
                    f"uplink_this_round={bits_this_round/1e6:.3f} Mbits "
                    f"uplink_total={total_uplink_bits/1e6:.3f} Mbits"
                )

        rewards_ma = moving_average(round_rewards_mean, ma_window)
        return {
            "algo": algo,
            "rounds": fed_cfg.rounds,
            "hidden": hidden,
            "comm_period": comm_period,
            "uplink_total_bits": int(total_uplink_bits),
            "uplink_total_bits_to_target": None if total_uplink_bits_to_target is None else int(total_uplink_bits_to_target),
            "rounds_to_target": rounds_to_target,
            "env_steps_total": int(env_steps_total),
            "env_steps_to_target": None if env_steps_to_target is None else int(env_steps_to_target),
            "auc_rewards_ma": auc(rewards_ma),
            "last_reward_ma": float(rewards_ma[-1]) if rewards_ma else 0.0,
        }

    elif algo == "fl_raw":
        # ---------------- FL RAW (dados brutos) ----------------
        server = CentralDQNServer(env.obs_dim, env.n_actions, hidden, dqn_cfg, device, seed=seed + 999)

        collectors = [
            CollectorAgent(env.obs_dim, env.n_actions, hidden, dqn_cfg, device, seed=seed + 1000 + l)
            for l in range(env_cfg.L)
        ]

        bits_per_transition = bits_raw_transition(env.obs_dim)

        for t_round in range(fed_cfg.rounds):
            # broadcast CPU->AP (não contamos)
            theta = server.get_param_vector()
            for c in collectors:
                c.set_param_vector(theta)

            obs_all = env.reset()
            local_round_rewards = []

            # coleta de dados brutos (sem treino local)
            for _ in range(fed_cfg.local_episodes_per_round):
                done = False
                while not done:
                    actions = [collectors[l].act(obs_all[l]) for l in range(env_cfg.L)]
                    obs2_all, rewards_all, done, _info = env.step(actions)

                    for l in range(env_cfg.L):
                        collectors[l].store_transition(obs_all[l], actions[l], rewards_all[l], obs2_all[l], done)

                    obs_all = obs2_all
                    local_round_rewards.append(float(np.mean(rewards_all)))

                    env_steps_total += 1

            # uplink: enviar transições brutas AP->CPU
            bits_this_round = 0
            all_transitions = []
            for c in collectors:
                trans = c.pop_all_transitions()
                all_transitions.extend(trans)
                bits_this_round += len(trans) * bits_per_transition

            total_uplink_bits += bits_this_round

            # CPU ingere e treina centralmente
            server.ingest_transitions(all_transitions)
            server.train_steps(server_train_steps_per_round)

            rr = float(np.mean(local_round_rewards)) if local_round_rewards else 0.0
            round_rewards_mean.append(rr)

            rewards_ma = moving_average(round_rewards_mean, ma_window)
            r_ma = rewards_ma[-1] if rewards_ma else 0.0

            if rounds_to_target is None and r_ma >= target_reward:
                rounds_to_target = t_round + 1
                total_uplink_bits_to_target = total_uplink_bits
                env_steps_to_target = env_steps_total

            if log_every > 0 and ((t_round + 1) % log_every == 0):
                print(
                    f"[fl_raw] round {t_round+1:04d}/{fed_cfg.rounds} "
                    f"reward_mean={rr:.3f} reward_MA({ma_window})={r_ma:.3f} "
                    f"uplink_this_round={bits_this_round/1e6:.3f} Mbits "
                    f"uplink_total={total_uplink_bits/1e6:.3f} Mbits"
                )

        rewards_ma = moving_average(round_rewards_mean, ma_window)
        return {
            "algo": "fl_raw",
            "rounds": fed_cfg.rounds,
            "hidden": hidden,
            "comm_period": None,
            "uplink_total_bits": int(total_uplink_bits),
            "uplink_total_bits_to_target": None if total_uplink_bits_to_target is None else int(total_uplink_bits_to_target),
            "rounds_to_target": rounds_to_target,
            "env_steps_total": int(env_steps_total),
            "env_steps_to_target": None if env_steps_to_target is None else int(env_steps_to_target),
            "auc_rewards_ma": auc(rewards_ma),
            "last_reward_ma": float(rewards_ma[-1]) if rewards_ma else 0.0,
        }

    else:
        raise ValueError("algo inválido")


def print_summary(res: Dict, target_reward: float):
    print("\n==================== RESULTADOS ====================")
    print(f"Algo: {res['algo']}")
    print(f"hidden: {res['hidden']}")
    if res.get("comm_period", None) is not None:
        print(f"comm_period (FRL): {res['comm_period']}")
    print(f"Rounds: {res['rounds']}")
    print(f"Rounds-to-target (MA >= {target_reward}): {res['rounds_to_target']}")
    print(f"Env steps total: {res['env_steps_total']}")
    print(f"Env steps to target: {res['env_steps_to_target']}")
    print(f"Uplink total (AP->CPU): {res['uplink_total_bits']/1e6:.3f} Mbits")
    if res["uplink_total_bits_to_target"] is None:
        print("Uplink até target: None (não convergiu)")
    else:
        print(f"Uplink até target: {res['uplink_total_bits_to_target']/1e6:.3f} Mbits")
    print(f"AUC(reward_ma): {res['auc_rewards_ma']:.3f}")
    print(f"Last reward_MA: {res['last_reward_ma']:.3f}")


def main():
    ap = argparse.ArgumentParser()

    # single run or compare
    ap.add_argument("--compare", action="store_true", help="Roda FRL (qgradual) e FL raw e compara")
    ap.add_argument("--algo", type=str, default="qgradual", choices=["qgradual", "fedavg", "fl_raw"])

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

    # modelos/comm
    ap.add_argument("--hidden", type=int, default=64, help="Tamanho do hidden do QNetwork (impacta uplink do delta)")
    ap.add_argument("--comm_period", type=int, default=3, help="FRL: envia delta a cada C rounds (sem compressão)")

    # Só para fl_raw: quantos updates DQN a CPU faz por round com os dados recebidos
    ap.add_argument("--server_train_steps_per_round", type=int, default=200)

    # DQN knobs
    ap.add_argument("--start_learn", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eps_decay_steps", type=int, default=8000)

    ap.add_argument("--log_every", type=int, default=20)

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

    if args.compare:
        print("\n=== Rodando FRL-QGradual ===")
        res_frl = run_experiment(
            algo="qgradual",
            seed=args.seed,
            env_cfg=env_cfg,
            dqn_cfg=dqn_cfg,
            fed_cfg=fed_cfg,
            device=device,
            hidden=args.hidden,
            comm_period=args.comm_period,
            target_reward=args.target_reward,
            ma_window=args.ma_window,
            server_train_steps_per_round=args.server_train_steps_per_round,
            log_every=args.log_every,
        )
        print_summary(res_frl, args.target_reward)

        print("\n=== Rodando FL clássico (dados brutos) ===")
        res_fl = run_experiment(
            algo="fl_raw",
            seed=args.seed,
            env_cfg=env_cfg,
            dqn_cfg=dqn_cfg,
            fed_cfg=fed_cfg,
            device=device,
            hidden=args.hidden,
            comm_period=args.comm_period,  # não usado no fl_raw
            target_reward=args.target_reward,
            ma_window=args.ma_window,
            server_train_steps_per_round=args.server_train_steps_per_round,
            log_every=args.log_every,
        )
        print_summary(res_fl, args.target_reward)

        # comparação direta
        print("\n==================== COMPARAÇÃO ====================")
        # Uplink
        if res_frl["uplink_total_bits_to_target"] is not None and res_fl["uplink_total_bits_to_target"] is not None:
            ratio = res_frl["uplink_total_bits_to_target"] / max(1, res_fl["uplink_total_bits_to_target"])
            print(f"Uplink até target (FRL / FL_raw): {ratio:.3f}x  (menor é melhor)")
        else:
            print("Uplink até target: não disponível para um dos métodos (não convergiu).")

        # Velocidade por rounds
        if res_frl["rounds_to_target"] is not None and res_fl["rounds_to_target"] is not None:
            print(f"Rounds-to-target: FRL={res_frl['rounds_to_target']} | FL_raw={res_fl['rounds_to_target']} (menor é melhor)")
        else:
            print("Rounds-to-target: não disponível para um dos métodos (não convergiu).")

        # Velocidade por env steps
        if res_frl["env_steps_to_target"] is not None and res_fl["env_steps_to_target"] is not None:
            print(f"Env-steps-to-target: FRL={res_frl['env_steps_to_target']} | FL_raw={res_fl['env_steps_to_target']} (menor é melhor)")
        else:
            print("Env-steps-to-target: não disponível para um dos métodos (não convergiu).")

        print("\nComandos exemplo:")
        print("  # Comparar os dois automaticamente:")
        print("  python frl_qgradual_cellfree.py --compare --hidden 64 --comm_period 3")
        print("  # Rodar só FRL-QGradual:")
        print("  python frl_qgradual_cellfree.py --algo qgradual --hidden 64 --comm_period 3")
        print("  # Rodar só FL clássico (dados brutos):")
        print("  python frl_qgradual_cellfree.py --algo fl_raw --hidden 64 --server_train_steps_per_round 200")

    else:
        res = run_experiment(
            algo=args.algo,
            seed=args.seed,
            env_cfg=env_cfg,
            dqn_cfg=dqn_cfg,
            fed_cfg=fed_cfg,
            device=device,
            hidden=args.hidden,
            comm_period=args.comm_period,
            target_reward=args.target_reward,
            ma_window=args.ma_window,
            server_train_steps_per_round=args.server_train_steps_per_round,
            log_every=args.log_every,
        )
        print_summary(res, args.target_reward)

        print("\nComandos sugeridos:")
        print("  # FRL-QGradual (uplink delta a cada comm_period rounds):")
        print("  python frl_qgradual_cellfree.py --algo qgradual --hidden 64 --comm_period 3")
        print("  # FL clássico com dados brutos:")
        print("  python frl_qgradual_cellfree.py --algo fl_raw --hidden 64 --server_train_steps_per_round 200")


if __name__ == "__main__":
    main()
