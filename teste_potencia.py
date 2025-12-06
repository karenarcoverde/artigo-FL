import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# Se quiser ver os logs de treino por rodada, mude para True
VERBOSE_TRAIN = False

# -----------------------------------------
# Ações do agente: frações de potência
# 0 -> 0.0 * Pmax
# 1 -> 0.5 * Pmax
# 2 -> 1.0 * Pmax
# -----------------------------------------
ACTION_LEVELS = np.array([0.0, 0.5, 1.0], dtype=float)


# =========================================
# 1. Métricas
# =========================================

def taxa_media_por_usuario(rates: np.ndarray) -> float:
    """
    Calcula a taxa média por usuário:
    - rates: array 2D shape (T, K)
             T = número de slots
             K = número de UEs
    Retorna um escalar (média global).
    """
    return float(np.mean(rates))


def ganho_percentual_taxa(rates_simple: np.ndarray,
                          rates_robust: np.ndarray) -> float:
    """
    Calcula o ganho percentual de taxa do cenário robusto (FRL)
    em relação ao cenário simples (baseline).

    Retorna (ganho_em_%, R_simple, R_robust)
    """
    R_simple = taxa_media_por_usuario(rates_simple)
    R_robust = taxa_media_por_usuario(rates_robust)

    if R_simple == 0:
        raise ValueError("Taxa média do cenário simples é zero; "
                         "não dá para calcular ganho percentual.")

    ganho = (R_robust - R_simple) / R_simple * 100.0
    return ganho, R_simple, R_robust


# =========================================
# 2. Agente RL (bandit) + AP + Servidor
# =========================================

class RLBanditAgent:
    """
    Agente bandit simples:
    - Sem estado (stateless): aprende qual ação (fração de potência)
      tende a dar maior recompensa média.
    - Atualização por média incremental.
    """
    def __init__(self,
                 n_actions: int = 3,
                 epsilon: float = 0.1,
                 rng: Optional[np.random.Generator] = None):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = np.zeros(n_actions, dtype=float)  # estimativa de valor por ação
        self.N = np.zeros(n_actions, dtype=float)  # contagem de vezes que cada ação foi escolhida
        # RNG próprio do agente
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self) -> int:
        """
        Escolhe uma ação usando exploração/exploração (ε-greedy).
        Retorna o índice da ação.
        """
        # Exploração
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # Exploitation (greedy)
        return int(np.argmax(self.Q))

    def learn(self, trajectories: List[Dict[str, Any]]):
        """
        Atualiza Q(a) com base nas transições:
        trajectories: lista de dicts {"action": a, "reward": r}
        """
        for tr in trajectories:
            a = tr["action"]
            r = tr["reward"]
            self.N[a] += 1.0
            alpha = 1.0 / self.N[a]  # passo decrescente
            self.Q[a] += alpha * (r - self.Q[a])

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Retorna parâmetros "treináveis" para FRL.
        """
        return {
            "Q": self.Q.copy(),
            "N": self.N.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        """
        Define parâmetros vindos do servidor (modelo global).
        """
        self.Q = params["Q"].copy()
        self.N = params["N"].copy()


@dataclass
class AccessPoint:
    ap_id: int
    position: np.ndarray
    rl_agent: RLBanditAgent


@dataclass
class CentralServer:
    """
    Servidor central que agrega os modelos dos APs (FRL / FedAvg / QGradual).
    """
    model_params: Dict[str, np.ndarray] = None
    quality_scores: Dict[int, float] = None  # para QGradual
    alpha_q: float = 0.3                     # passo da atualização QGradual

    # --------- FedAvg base ---------
    def _fedavg(self,
                ap_models: Dict[int, Dict[str, np.ndarray]],
                weights: Dict[int, float]):
        """
        Implementação base do FedAvg ponderado.
        """
        if not ap_models:
            return

        first_ap_id = next(iter(ap_models.keys()))
        keys = ap_models[first_ap_id].keys()

        aggregated = {
            k: np.zeros_like(ap_models[first_ap_id][k], dtype=float)
            for k in keys
        }

        total_w = sum(weights.values()) + 1e-12
        for ap_id, params in ap_models.items():
            w = weights[ap_id] / total_w
            for k in keys:
                aggregated[k] += w * params[k]

        self.model_params = aggregated

    # --------- FedAvg normal (sem QGradual) ---------
    def aggregate(self,
                  ap_models: Dict[int, Dict[str, np.ndarray]],
                  weights: Dict[int, float] = None):
        """
        Agregação tipo FedAvg "clássico":
        - Aqui NÃO usamos reward como peso.
        - Todos os APs têm o mesmo peso.
        """
        if weights is None:
            weights = {ap_id: 1.0 for ap_id in ap_models.keys()}
        self._fedavg(ap_models, weights)

    # --------- Agregação QGradual ---------
    def aggregate_qgradual(self,
                           ap_models: Dict[int, Dict[str, np.ndarray]],
                           rewards_accum: Dict[int, float]):
        """
        Agregação QGradual:
        - mantém score de qualidade q_m(t) para cada AP;
        - atualiza q_m(t) com base na recompensa normalizada da rodada;
        - usa q_m(t)^2 como peso no FedAvg (dá mais peso a APs bons).
        """
        if not ap_models:
            return

        # inicializa quality_scores se necessário
        if self.quality_scores is None:
            self.quality_scores = {ap_id: 1.0 for ap_id in rewards_accum.keys()}

        rewards = np.array(list(rewards_accum.values()), dtype=float)
        ap_ids = list(rewards_accum.keys())

        # normalização dos rewards da rodada
        r_min = rewards.min()
        r_max = rewards.max()
        if r_max - r_min < 1e-12:
            norm_rewards = np.ones_like(rewards)
        else:
            norm_rewards = (rewards - r_min) / (r_max - r_min + 1e-12)

        # q_m(t) = (1 - alpha_q)*q_m(t-1) + alpha_q * r_norm(t)
        for i, ap_id in enumerate(ap_ids):
            r_norm = float(norm_rewards[i])
            q_old = self.quality_scores.get(ap_id, 1.0)
            q_new = (1.0 - self.alpha_q) * q_old + self.alpha_q * r_norm
            self.quality_scores[ap_id] = q_new

        # usa q_m(t)^2 como pesos no FedAvg (reforça APs consistentemente bons)
        weights_q = {ap_id: self.quality_scores[ap_id] ** 2 for ap_id in ap_ids}
        self._fedavg(ap_models, weights_q)

    def broadcast(self, aps: List[AccessPoint]):
        """
        Envia o modelo global para todos os APs.
        """
        if self.model_params is None:
            return
        for ap in aps:
            ap.rl_agent.set_params(self.model_params)


# =========================================
# 3. Canal: pathloss (grande escala)
# =========================================

def compute_large_scale(ap_pos: np.ndarray,
                        ue_pos: np.ndarray,
                        alpha: float = 3.7) -> np.ndarray:
    """
    Calcula ganho de grande escala (pathloss ~ d^-alpha)
    ap_pos: (M, 2)
    ue_pos: (K, 2)
    Retorna: (M, K)
    """
    M = ap_pos.shape[0]
    K = ue_pos.shape[0]
    large_scale = np.zeros((M, K), dtype=float)

    for m in range(M):
        for k in range(K):
            d = np.linalg.norm(ap_pos[m] - ue_pos[k]) + 1e-3  # evita zero
            large_scale[m, k] = d ** (-alpha)

    return large_scale


# =========================================
# 4. Cenário simples (baseline)
# =========================================

def simulate_baseline(ap_pos: np.ndarray,
                      ue_pos: np.ndarray,
                      large_scale: np.ndarray,
                      T_eval: int,
                      p_max: float,
                      baseline_fraction: float,
                      noise_power: float,
                      seed: int) -> np.ndarray:
    """
    Simula o cenário cell-free simples (baseline):
    - todos os APs transmitem com fração fixa de Pmax para o UE ativo do slot.
    - UE ativo do slot é escolhido aleatoriamente.

    Retorna:
        rates_simple: array (T_eval, K)
    """
    rng = np.random.default_rng(seed)
    M, K = large_scale.shape
    rates_simple = np.zeros((T_eval, K), dtype=float)

    for t in range(T_eval):
        # desvanecimento de pequena escala Rayleigh
        h = (rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))) / math.sqrt(2.0)
        g = large_scale * (np.abs(h) ** 2)

        # escolhe um UE ativo aleatoriamente
        ue_idx = int(rng.integers(0, K))

        tx_power = baseline_fraction * p_max
        # potência recebida pelo UE ativo = soma de todos os APs
        S = tx_power * g[:, ue_idx].sum()
        snr = S / noise_power
        R = math.log2(1.0 + snr)

        rates_simple[t, ue_idx] = R

    return rates_simple


# =========================================
# 5. Treino FRL (cenário robusto - FedAvg "padrão")
# =========================================

def train_frl(ap_pos: np.ndarray,
              ue_pos: np.ndarray,
              large_scale: np.ndarray,
              num_rounds: int,
              steps_per_round: int,
              p_max: float,
              noise_power: float,
              ap_eps: np.ndarray,
              seed_train: int) -> List[AccessPoint]:
    """
    Treina APs como agentes RL (bandits) com agregação federada (FRL-FedAvg).

    - num_rounds: número de rodadas federadas
    - steps_per_round: número de slots por rodada
    - ap_eps: vetor com epsilon de cada AP (heterogêneo!)

    Retorna:
        lista de APs treinados (com seus agentes)
    """
    M, K = large_scale.shape

    # cria APs, cada um com seu agente bandit, epsilon e RNG próprios
    aps: List[AccessPoint] = []
    for m in range(M):
        rng_agent = np.random.default_rng(seed_train + 1000 + m)
        agent = RLBanditAgent(
            n_actions=len(ACTION_LEVELS),
            epsilon=float(ap_eps[m]),
            rng=rng_agent
        )
        aps.append(
            AccessPoint(
                ap_id=m,
                position=ap_pos[m],
                rl_agent=agent
            )
        )

    server = CentralServer()  # aqui usamos aggregate (FedAvg simples)
    rng = np.random.default_rng(seed_train)

    for rnd in range(num_rounds):
        trajectories_per_ap: Dict[int, List[Dict[str, Any]]] = {
            ap.ap_id: [] for ap in aps
        }
        rewards_accum: Dict[int, float] = {
            ap.ap_id: 0.0 for ap in aps
        }

        for step in range(steps_per_round):
            # canal de pequena escala Rayleigh
            h = (rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))) / math.sqrt(2.0)
            g = large_scale * (np.abs(h) ** 2)

            # escolhe um UE ativo aleatoriamente no slot
            ue_idx = int(rng.integers(0, K))

            # cada AP escolhe sua ação (fração de potência)
            actions_idx: Dict[int, int] = {}
            for ap in aps:
                a_idx = ap.rl_agent.act()
                actions_idx[ap.ap_id] = a_idx

            # --------- contribuições locais por AP ---------
            contribs: Dict[int, float] = {}
            S_total = 0.0
            for ap in aps:
                frac = ACTION_LEVELS[actions_idx[ap.ap_id]]
                contrib = frac * p_max * g[ap.ap_id, ue_idx]
                contribs[ap.ap_id] = contrib
                S_total += contrib

            snr = S_total / noise_power
            R_slot = math.log2(1.0 + snr)

            # recompensa local: proporcional à fração que cada AP contribui para S_total
            for ap in aps:
                if S_total > 0.0:
                    share = contribs[ap.ap_id] / (S_total + 1e-12)
                else:
                    share = 0.0
                reward_ap = R_slot * share

                trajectories_per_ap[ap.ap_id].append({
                    "action": actions_idx[ap.ap_id],
                    "reward": reward_ap,
                })
                rewards_accum[ap.ap_id] += reward_ap
            # ------------------------------------------------------

        # atualização local em cada AP
        for ap in aps:
            ap.rl_agent.learn(trajectories_per_ap[ap.ap_id])

        # agregação FedAvg COM PESOS IGUAIS (não usa rewards!)
        ap_models = {ap.ap_id: ap.rl_agent.get_params() for ap in aps}
        server.aggregate(ap_models)  # pesos uniformes
        server.broadcast(aps)

        if VERBOSE_TRAIN:
            media_reward = np.mean(list(rewards_accum.values())) / steps_per_round
            print(f"[FRL-FedAvg] Round {rnd+1}/{num_rounds} - reward médio por slot: {media_reward:.4f}")

    return aps


# =========================================
# 5b. Treino FRL-QGradual (cenário robusto "melhorado")
# =========================================

def train_frl_qgradual(ap_pos: np.ndarray,
                       ue_pos: np.ndarray,
                       large_scale: np.ndarray,
                       num_rounds: int,
                       steps_per_round: int,
                       p_max: float,
                       noise_power: float,
                       ap_eps: np.ndarray,
                       seed_train: int,
                       alpha_q: float = 0.3) -> List[AccessPoint]:
    """
    Treina APs como agentes RL (bandits) com agregação federada (FRL-QGradual).

    Mesma estrutura do train_frl, mas o servidor usa aggregate_qgradual.
    """
    M, K = large_scale.shape

    aps: List[AccessPoint] = []
    for m in range(M):
        rng_agent = np.random.default_rng(seed_train + 2000 + m)
        agent = RLBanditAgent(
            n_actions=len(ACTION_LEVELS),
            epsilon=float(ap_eps[m]),
            rng=rng_agent
        )
        aps.append(
            AccessPoint(
                ap_id=m,
                position=ap_pos[m],
                rl_agent=agent
            )
        )

    server = CentralServer(alpha_q=alpha_q)
    rng = np.random.default_rng(seed_train)

    for rnd in range(num_rounds):
        trajectories_per_ap: Dict[int, List[Dict[str, Any]]] = {
            ap.ap_id: [] for ap in aps
        }
        rewards_accum: Dict[int, float] = {
            ap.ap_id: 0.0 for ap in aps
        }

        for step in range(steps_per_round):
            h = (rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))) / math.sqrt(2.0)
            g = large_scale * (np.abs(h) ** 2)

            ue_idx = int(rng.integers(0, K))

            actions_idx: Dict[int, int] = {}
            for ap in aps:
                a_idx = ap.rl_agent.act()
                actions_idx[ap.ap_id] = a_idx

            # --------- MESMA LÓGICA DE RECOMPENSA LOCAL ---------
            contribs: Dict[int, float] = {}
            S_total = 0.0
            for ap in aps:
                frac = ACTION_LEVELS[actions_idx[ap.ap_id]]
                contrib = frac * p_max * g[ap.ap_id, ue_idx]
                contribs[ap.ap_id] = contrib
                S_total += contrib

            snr = S_total / noise_power
            R_slot = math.log2(1.0 + snr)

            for ap in aps:
                if S_total > 0.0:
                    share = contribs[ap.ap_id] / (S_total + 1e-12)
                else:
                    share = 0.0
                reward_ap = R_slot * share

                trajectories_per_ap[ap.ap_id].append({
                    "action": actions_idx[ap.ap_id],
                    "reward": reward_ap,
                })
                rewards_accum[ap.ap_id] += reward_ap
            # ----------------------------------------------------

        for ap in aps:
            ap.rl_agent.learn(trajectories_per_ap[ap.ap_id])

        ap_models = {ap.ap_id: ap.rl_agent.get_params() for ap in aps}
        # agora usamos QGradual (com pesos baseados em qualidade acumulada)
        server.aggregate_qgradual(ap_models, rewards_accum)
        server.broadcast(aps)

        if VERBOSE_TRAIN:
            media_reward = np.mean(list(rewards_accum.values())) / steps_per_round
            print(f"[FRL-QGradual] Round {rnd+1}/{num_rounds} - reward médio por slot: {media_reward:.4f}")

    return aps


# =========================================
# 6. Avaliação do cenário robusto (FRL)
# =========================================

def evaluate_frl(aps: List[AccessPoint],
                 ap_pos: np.ndarray,
                 ue_pos: np.ndarray,
                 large_scale: np.ndarray,
                 T_eval: int,
                 p_max: float,
                 noise_power: float,
                 seed_eval: int) -> np.ndarray:
    """
    Avalia o desempenho dos APs treinados com FRL:
    - coloca epsilon = 0 (política puramente greedy)
    - roda T_eval slots, UE ativo aleatório
    - retorna matriz (T_eval, K) de taxas
    """
    M, K = large_scale.shape
    rates_robust = np.zeros((T_eval, K), dtype=float)
    rng = np.random.default_rng(seed_eval)

    # guarda epsilons antigos e zera para avaliação (exploit-only)
    old_eps = [ap.rl_agent.epsilon for ap in aps]
    for ap in aps:
        ap.rl_agent.epsilon = 0.0

    for t in range(T_eval):
        h = (rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))) / math.sqrt(2.0)
        g = large_scale * (np.abs(h) ** 2)

        ue_idx = int(rng.integers(0, K))

        S = 0.0
        for ap in aps:
            a_idx = ap.rl_agent.act()
            frac = ACTION_LEVELS[a_idx]
            S += frac * p_max * g[ap.ap_id, ue_idx]

        snr = S / noise_power
        R = math.log2(1.0 + snr)
        rates_robust[t, ue_idx] = R

    # restaura epsilons
    for ap, eps in zip(aps, old_eps):
        ap.rl_agent.epsilon = eps

    return rates_robust


# =========================================
# 7. Main: rodar múltiplas AVALIAÇÕES e comparar
# =========================================

if __name__ == "__main__":
    # Parâmetros da rede
    M = 4            # número de APs
    K = 6            # número de UEs
    area_size = 500  # metros
    p_max = 1.0      # W
    noise_power = 1e-9

    # Heterogeneidade no CANAL
    ap_quality = np.array([0.05, 0.2, 1.0, 4.0], dtype=float)

    # Heterogeneidade no APRENDIZADO (ε)
    ap_eps = np.array([0.9, 0.8, 0.3, 0.05], dtype=float)

    # Parâmetros de simulação
    T_eval = 5000            # maior para reduzir variância por avaliação
    baseline_fraction = 0.3  # fração fixa de Pmax em todos os APs
    num_rounds = 30
    steps_per_round = 100

    # Número de AVALIAÇÕES independentes (canal/UE diferentes)
    N_EVAL = 30

    # ---------- Topologia fixa ----------
    rng_top = np.random.default_rng(42)  # seed fixa só para topologia
    ap_pos = rng_top.uniform(0.0, area_size, size=(M, 2))
    ue_pos = rng_top.uniform(0.0, area_size, size=(K, 2))
    large_scale = compute_large_scale(ap_pos, ue_pos, alpha=3.7)
    large_scale = large_scale * ap_quality[:, None]

    # ---------- TREINO ÚNICO dos modelos ----------
    seed_train_fedavg = 20000
    seed_train_qgrad  = 30000

    aps_fedavg = train_frl(
        ap_pos=ap_pos,
        ue_pos=ue_pos,
        large_scale=large_scale,
        num_rounds=num_rounds,
        steps_per_round=steps_per_round,
        p_max=p_max,
        noise_power=noise_power,
        ap_eps=ap_eps,
        seed_train=seed_train_fedavg
    )

    aps_qgradual = train_frl_qgradual(
        ap_pos=ap_pos,
        ue_pos=ue_pos,
        large_scale=large_scale,
        num_rounds=num_rounds,
        steps_per_round=steps_per_round,
        p_max=p_max,
        noise_power=noise_power,
        ap_eps=ap_eps,
        seed_train=seed_train_qgrad,
        alpha_q=0.3
    )

    ganhos_fedavg_list = []
    ganhos_qgradual_list = []
    ganhos_qgradual_fedavg_list = []

    # ---------- VÁRIAS AVALIAÇÕES (canal/UE aleatórios) ----------
    for exp in range(N_EVAL):
        seed_eval = 10000 + exp  # muda só o canal/UE

        # Baseline
        rates_simple = simulate_baseline(
            ap_pos=ap_pos,
            ue_pos=ue_pos,
            large_scale=large_scale,
            T_eval=T_eval,
            p_max=p_max,
            baseline_fraction=baseline_fraction,
            noise_power=noise_power,
            seed=seed_eval
        )

        # FedAvg
        rates_fedavg = evaluate_frl(
            aps=aps_fedavg,
            ap_pos=ap_pos,
            ue_pos=ue_pos,
            large_scale=large_scale,
            T_eval=T_eval,
            p_max=p_max,
            noise_power=noise_power,
            seed_eval=seed_eval
        )

        # QGradual
        rates_qgradual = evaluate_frl(
            aps=aps_qgradual,
            ap_pos=ap_pos,
            ue_pos=ue_pos,
            large_scale=large_scale,
            T_eval=T_eval,
            p_max=p_max,
            noise_power=noise_power,
            seed_eval=seed_eval
        )

        # Ganhos
        ganho_fedavg, R_simple, R_fedavg = ganho_percentual_taxa(rates_simple, rates_fedavg)
        ganho_qgradual_base, _, R_qgradual = ganho_percentual_taxa(rates_simple, rates_qgradual)
        ganho_qgradual_fedavg = (R_qgradual - R_fedavg) / (R_fedavg + 1e-12) * 100.0

        ganhos_fedavg_list.append(ganho_fedavg)
        ganhos_qgradual_list.append(ganho_qgradual_base)
        ganhos_qgradual_fedavg_list.append(ganho_qgradual_fedavg)

        print(f"[Eval {exp+1}/{N_EVAL}] "
              f"G_fedavg={ganho_fedavg:.2f}% | "
              f"G_qgrad_base={ganho_qgradual_base:.2f}% | "
              f"G_qgrad_vs_fedavg={ganho_qgradual_fedavg:.2f}%")

    # ---------- Estatísticas finais ----------
    print("\n=======================================")
    print("   ESTATÍSTICA SOBRE N AVALIAÇÕES")
    print("=======================================")
    print("FedAvg vs baseline:  média = %.2f%%, std = %.2f"
          % (np.mean(ganhos_fedavg_list), np.std(ganhos_fedavg_list)))
    print("QGradual vs baseline: média = %.2f%%, std = %.2f"
          % (np.mean(ganhos_qgradual_list), np.std(ganhos_qgradual_list)))
    print("QGradual vs FedAvg:   média = %.2f%%, std = %.2f"
          % (np.mean(ganhos_qgradual_fedavg_list), np.std(ganhos_qgradual_fedavg_list)))
