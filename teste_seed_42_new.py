import os, random, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from collections import defaultdict
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# ============================================================
# ===================== FAIRNESS (JAIN) ======================
# ============================================================
def jain_index(x):
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0
    s = x.sum()
    s2 = np.sum(x**2)
    if s2 == 0:
        return 0.0
    return (s**2) / (n * s2)

# ============================================================
# ===================== SEEDS / DETERMINISMO =================
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# ============================================================
# ===================== CONFIGS (ARTIGO) =====================
# ============================================================
NUM_AP = 16
K_USERS = 20
TOTAL_USERS = K_USERS

EPOCHS_LOCAL  = 5
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64
VAL_SAMPLES = 5000

ALPHA_LEVEL = 0.7

USE_ALL_USERS_PER_AP = False
K_PER_AP = 4

EPS = 0.05
TEMP_SEL = 0.8
FAIR_FRACTION = 0.25

# ============================================================
# ===================== RL: LIGAR / MODO =====================
# ============================================================
USE_RL = True
RL_REWARD_MODE = "cluster_link"  # "cluster_link" = reward combinado do cluster (0..1)
LINK_QUALITY_GAMMA = 0.25        # mistura Q + link na seleção (por AP)

# ============================================================
# ============== COMBINAÇÃO DE LINK DO CLUSTER ===============
# ============================================================
# "max"  : max(lq_ap_ue) / 1
# "sum"  : sum(lq_ap_ue) / NUM_AP
# "mrc"  : sqrt(sum(lq^2)) / sqrt(NUM_AP)   (MRC-like)
CLUSTER_COMBINE_MODE = "mrc"

def combine_cluster_link(lq_list, mode="mrc", num_ap=16):
    if len(lq_list) == 0:
        return 0.0
    lq = np.array(lq_list, dtype=np.float32)

    if mode == "max":
        comb = float(np.max(lq))               # já em [0,1]
        return comb

    if mode == "sum":
        comb = float(np.sum(lq))               # em [0, len(lq)]
        return comb / max(1.0, float(num_ap))  # normaliza p/ [0,1]

    # default: "mrc"
    comb = float(np.sqrt(np.sum(lq**2)))       # em [0, sqrt(len(lq))]
    return comb / max(1e-9, float(np.sqrt(num_ap)))  # p/ [0,1]

# ============================================================
# ===================== BASELINE: SEM RL =====================
# ============================================================
BASELINE_MODE = "channel"  # "random" | "linkfair" | "channel"
# (Se USE_RL=True, BASELINE_MODE não é usado.)

# ============================================================
# ============== “CELL-FREE” (APs x UEs) =====================
# ============================================================
ap_users = {ap: list(range(TOTAL_USERS)) for ap in range(NUM_AP)}

PATHLOSS_EXP = 3.7
MIN_DIST = 0.02
SHADOWING_STD_DB = 6.0

def generate_beta_matrix(num_ap, num_ue, area_side=1.0,
                         pathloss_exp=3.7, min_dist=0.02, shadow_std_db=6.0, seed=42):
    rng = np.random.default_rng(seed)
    ap_pos = rng.uniform(0.0, area_side, size=(num_ap, 2))
    ue_pos = rng.uniform(0.0, area_side, size=(num_ue, 2))

    diff = ap_pos[:, None, :] - ue_pos[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    d = np.maximum(d, min_dist)

    pl = d ** (-pathloss_exp)
    shadow_db = rng.normal(0.0, shadow_std_db, size=pl.shape)
    shadow_lin = 10 ** (shadow_db / 10.0)

    beta = pl * shadow_lin
    return beta

beta_mk = generate_beta_matrix(
    num_ap=NUM_AP,
    num_ue=TOTAL_USERS,
    area_side=1.0,
    pathloss_exp=PATHLOSS_EXP,
    min_dist=MIN_DIST,
    shadow_std_db=SHADOWING_STD_DB,
    seed=SEED
)

# link_quality[ap][u] em [0,1]
link_quality = {ap: {} for ap in range(NUM_AP)}
for ap in range(NUM_AP):
    ues = ap_users[ap]
    b = np.array([beta_mk[ap, u] for u in ues], dtype=np.float32)
    bmin, bmax = float(np.min(b)), float(np.max(b))
    if (bmax - bmin) < 1e-12:
        for u in ues:
            link_quality[ap][u] = 0.5
    else:
        for u, val in zip(ues, b):
            link_quality[ap][u] = float((val - bmin) / (bmax - bmin))

# ============================================================
# ===================== DADOS (IID) ==========================
# ============================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = y_train.squeeze().astype(np.int32)
y_test  = y_test.squeeze().astype(np.int32)

TOTAL_TRAIN = len(x_train)
if TOTAL_TRAIN % TOTAL_USERS != 0:
    raise ValueError(
        f"|D|={TOTAL_TRAIN} não é divisível por K={TOTAL_USERS}. "
        f"No artigo, K∈{{20,40}} divide exatamente 50.000."
    )

SAMPLES_PER_USER = TOTAL_TRAIN // TOTAL_USERS

x_val = x_test[:VAL_SAMPLES]
y_val = y_test[:VAL_SAMPLES]

def make_iid_splits_by_class_full(x, y, K, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = int(np.max(y)) + 1

    per_user_idx = [[] for _ in range(K)]
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        for i, idx in enumerate(idx_c):
            per_user_idx[i % K].append(idx)

    user_data_local = []
    sizes = []
    for u in range(K):
        idx_u = np.array(per_user_idx[u], dtype=np.int32)
        rng.shuffle(idx_u)
        user_data_local.append((x[idx_u], y[idx_u]))
        sizes.append(len(idx_u))

    if len(set(sizes)) != 1:
        raise RuntimeError(f"Partição IID não ficou igual: sizes={sizes}")

    return user_data_local

user_data = make_iid_splits_by_class_full(x_train, y_train, K=TOTAL_USERS, seed=SEED)

print("\n===== DADOS (IID igual ao artigo) =====")
print(f"|D| (total treino CIFAR-10)  = {TOTAL_TRAIN}")
print(f"K (usuários)                 = {TOTAL_USERS}")
print(f"|D_j| por usuário             = {SAMPLES_PER_USER}  (|D|/K)")
print(f"rho_j                         = 1/K = {1.0/TOTAL_USERS:.4f}")
print("======================================\n")

# ============================================================
# ===================== MODELO ===============================
# ============================================================
def build_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# ===================== “BANDIT” BASE ========================
# ============================================================
class BanditSelector:
    def __init__(self, ap_to_users):
        self.n = defaultdict(lambda: defaultdict(int))
        self.last_round = defaultdict(lambda: defaultdict(int))
        self.ap_to_users = ap_to_users

    def mark_participation(self, ap, uid, round_idx):
        self.n[ap][uid] += 1
        self.last_round[ap][uid] = round_idx

# ============================================================
# ===================== RL (BANDIT COM Q) ====================
# ============================================================
class RLBanditSelector(BanditSelector):
    """
    Bandit com Q(ap, uid).
    Aqui vamos atualizar Q usando reward do CLUSTER (0..1).
    """
    def __init__(self, ap_to_users):
        super().__init__(ap_to_users)
        self.Q = defaultdict(lambda: defaultdict(float))

    def update_q(self, ap, uid, reward, lr=ALPHA_LEVEL):
        q_old = self.Q[ap][uid]
        self.Q[ap][uid] = (1.0 - lr) * q_old + lr * float(reward)

    def get_q(self, ap, uid):
        return float(self.Q[ap][uid])

# ============================================================
# ===================== FEDAVG ===============================
# ============================================================
def fedavg(weights_list, sizes):
    N = float(np.sum(sizes))
    if N <= 0 or len(weights_list) == 0:
        return None
    new_weights = []
    for layer_ws in zip(*weights_list):
        stack = np.stack(layer_ws, axis=0)
        coeffs = (np.array(sizes)/N).reshape((-1,) + (1,)*(stack.ndim-1))
        new_weights.append(np.sum(coeffs * stack, axis=0))
    return new_weights

# ============================================================
# ===================== SELEÇÃO TOP-K ========================
# ============================================================
def softmax_scores(vec, temp=1.0, eps=1e-9):
    v = np.array(vec, dtype=np.float32)
    v = v - np.max(v)
    sm = np.exp(v / max(temp, 1e-3))
    return sm / (np.sum(sm) + eps)

def select_topk_no_prefilter_baseline(ap, bandit, K, round_idx,
                                      candidates, link_q_dict,
                                      temp=TEMP_SEL, eps=EPS,
                                      fair_fraction=FAIR_FRACTION,
                                      gamma_link=LINK_QUALITY_GAMMA):
    users = candidates
    if len(users) == 0:
        return []

    K = min(int(K), len(users))
    if K == len(users):
        return users[:]

    # ---------------------------------------------------------
    # (RL) seleção por Q(ap,uid) aprendida
    # score = Q + gamma_link * link_quality(ap,uid)
    # ---------------------------------------------------------
    if USE_RL:
        if np.random.rand() < eps:
            return list(np.random.choice(users, size=K, replace=False))

        q  = np.array([bandit.get_q(ap, u) for u in users], dtype=np.float32)
        lq = np.array([link_q_dict.get(u, 0.0) for u in users], dtype=np.float32)
        scores = q + gamma_link * lq

        tie = np.array([bandit.n[ap][u] for u in users], dtype=np.int32)
        order = np.lexsort((tie, -scores))
        return [users[i] for i in order[:K]]

    # ---------------------------------------------------------
    # baselines (se USE_RL=False)
    # ---------------------------------------------------------
    if BASELINE_MODE == "random":
        return list(np.random.choice(users, size=K, replace=False))

    if BASELINE_MODE == "channel":
        if np.random.rand() < eps:
            return list(np.random.choice(users, size=K, replace=False))

        lq = np.array([link_q_dict.get(u, 0.0) for u in users], dtype=np.float32)
        tie = np.array([bandit.n[ap][u] for u in users], dtype=np.int32)
        order = np.lexsort((tie, -lq))
        return [users[i] for i in order[:K]]

    # linkfair (age + link)
    if np.random.rand() < eps:
        return list(np.random.choice(users, size=K, replace=False))

    ages = np.array(
        [round_idx - bandit.last_round[ap][u] if bandit.last_round[ap][u] else round_idx + 1
         for u in users], dtype=np.float32
    )
    if float(np.max(ages)) > 0:
        ages = ages / (float(np.max(ages)) + 1e-9)

    lq = np.array([link_q_dict.get(u, 0.0) for u in users], dtype=np.float32)
    fairness_boost = 0.10
    scores = fairness_boost * ages + gamma_link * lq

    k_fair = max(1, int(np.ceil(fair_fraction * K)))
    tie = np.array([bandit.n[ap][u] for u in users], dtype=np.int32)
    rank_fair = np.lexsort((tie, -ages))
    fair_candidates = [users[i] for i in rank_fair[:k_fair]]

    remaining = K - len(fair_candidates)
    if remaining > 0:
        idx_map = {u:i for i,u in enumerate(users)}
        chosen_set = set(fair_candidates)
        users_left = [u for u in users if u not in chosen_set]
        probs_left = softmax_scores([scores[idx_map[u]] for u in users_left], temp=temp)
        chosen = list(np.random.choice(users_left, size=remaining, replace=False, p=probs_left))
        return fair_candidates + chosen

    return fair_candidates[:K]

# ============================================================
# === NOVO: SELEÇÃO "CLUSTER" (UE PODE REPETIR EM VÁRIOS APs) =
# ============================================================
def select_cluster_by_ap_round(bandit, round_idx, all_ues):
    """
    Cada AP seleciona seus K_PER_AP UEs (independente).
    UE pode ser escolhido por múltiplos APs no mesmo round (cluster).
    """
    selected_by_ap = {}
    all_ues = list(all_ues)

    for ap in range(NUM_AP):
        candidates_ap = all_ues[:]  # TODOS os UEs são candidatos p/ cada AP

        if USE_ALL_USERS_PER_AP:
            chosen = candidates_ap
        else:
            chosen = select_topk_no_prefilter_baseline(
                ap=ap,
                bandit=bandit,
                K=min(K_PER_AP, len(candidates_ap)),
                round_idx=round_idx,
                candidates=candidates_ap,
                link_q_dict=link_quality[ap],
                temp=TEMP_SEL,
                eps=EPS,
                fair_fraction=FAIR_FRACTION,
                gamma_link=LINK_QUALITY_GAMMA
            )

        chosen = list(dict.fromkeys(chosen))
        selected_by_ap[ap] = chosen

    return selected_by_ap

# ============================================================
# =======  MÉTRICAS DE FLOPs / PARÂMETROS / TRÁFEGO  =========
# ============================================================
def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

def inference_flops_tf(model, input_shape=(1,32,32,3), dtype=tf.float32):
    @tf.function
    def forward(x):
        return model(x, training=False)

    concrete = forward.get_concrete_function(tf.TensorSpec(input_shape, dtype))
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

def train_flops_per_sample_from_infer(infer_flops):
    return infer_flops * 2.5

def pretty_num(x):
    for u in ["", "k", "M", "G", "T", "P"]:
        if abs(x) < 1000:
            return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{x:.2f}E"

# ============================================================
# ===================== LOOP FEDERADO ========================
# ============================================================
bandit = RLBanditSelector(ap_users) if USE_RL else BanditSelector(ap_users)
global_model = build_model(num_classes=10)

n_params = count_params(global_model)
infer_flops = inference_flops_tf(global_model)
train_flops_per_sample = train_flops_per_sample_from_infer(infer_flops)

BYTES_PER_PARAM = 4
BYTES_PER_UE_PER_ROUND = 2 * n_params * BYTES_PER_PARAM  # down + up (UE<->servidor)

print("\n===== CONFIG (CELL-FREE CLUSTER) =====")
print(f"M (APs) = {NUM_AP}")
print(f"K (UEs) = {TOTAL_USERS}")
print(f"|D_j| (amostras por UE) = {SAMPLES_PER_USER}  (igual ao artigo)")
print(f"K por AP = {'ALL' if USE_ALL_USERS_PER_AP else K_PER_AP}")
print(f"USE_RL = {USE_RL} | RL_REWARD_MODE = {RL_REWARD_MODE}")
print(f"Cluster combine = {CLUSTER_COMBINE_MODE}")
print("=====================================\n")

print("===== MODELO =====")
print(f"Parâmetros: {n_params:,}")
print(f"FLOPs/amostra (forward): {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):  {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)\n")

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# fairness: conta "rounds em que UE treinou" (UE único por rodada)
ue_round_participations = np.zeros(TOTAL_USERS, dtype=np.int32)

# só para estatística: quantas vezes apareceu em (AP,UE)
ue_ap_instances = np.zeros(TOTAL_USERS, dtype=np.int32)

total_flops_all_rounds = 0.0
total_bytes_all_rounds = 0.0
total_unique_ues_updates = 0
total_ap_ue_instances = 0

ALL_UES = list(range(TOTAL_USERS))

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    # (1) cada AP escolhe seus UEs (cluster: UE pode repetir em vários APs)
    selected_by_ap = select_cluster_by_ap_round(bandit, r, ALL_UES)

    # (2) monta cluster_map: para cada UE, quais APs o selecionaram
    cluster_map = defaultdict(list)
    for ap, uids in selected_by_ap.items():
        for uid in uids:
            cluster_map[uid].append(ap)

    selected_union = sorted(cluster_map.keys())
    num_ues_unique_round = len(selected_union)
    total_selected_instances = sum(len(uids) for uids in selected_by_ap.values())

    # estatística de cluster
    cluster_sizes = [len(cluster_map[uid]) for uid in selected_union] if selected_union else [0]
    avg_cluster = float(np.mean(cluster_sizes)) if len(cluster_sizes) else 0.0

    # (3) treina CADA UE UMA VEZ (mesmo se aparecer em vários APs)
    local_weights = {}
    local_sizes = {}

    for uid in selected_union:
        local_model = build_model(num_classes=10)
        local_model.set_weights(global_model.get_weights())

        x_u, y_u = user_data[uid]
        local_model.fit(
            x_u, y_u,
            epochs=EPOCHS_LOCAL,
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=True
        )

        local_weights[uid] = local_model.get_weights()
        local_sizes[uid] = len(x_u)

        # fairness por rodada (UE treinou neste round)
        ue_round_participations[uid] += 1

        # conta instâncias AP×UE (estatística)
        ue_ap_instances[uid] += len(cluster_map[uid])

        # (4) reward do RL = link combinado do cluster
        if USE_RL and RL_REWARD_MODE == "cluster_link":
            lqs = [float(link_quality[ap].get(uid, 0.0)) for ap in cluster_map[uid]]
            reward_cluster = combine_cluster_link(
                lq_list=lqs,
                mode=CLUSTER_COMBINE_MODE,
                num_ap=NUM_AP
            )  # em [0,1]

            # atualiza Q(ap,uid) para TODOS os APs que serviram esse UE
            for ap in cluster_map[uid]:
                bandit.update_q(ap, uid, reward_cluster, lr=ALPHA_LEVEL)

    # marca participações (AP,UE) para desempate/fairness por AP
    for ap, uids in selected_by_ap.items():
        for uid in uids:
            bandit.mark_participation(ap, uid, r)

    # (5) agregação GLOBAL (FedAvg clássico) sobre UEs únicos do round
    weights_list = [local_weights[uid] for uid in selected_union]
    sizes_list   = [local_sizes[uid]   for uid in selected_union]

    new_global_weights = fedavg(weights_list, sizes_list)
    if new_global_weights is not None:
        global_model.set_weights(new_global_weights)

    # (6) custo (agora baseado em UEs únicos que realmente treinaram)
    flops_this_round = train_flops_per_sample * SAMPLES_PER_USER * EPOCHS_LOCAL * num_ues_unique_round
    bytes_this_round = BYTES_PER_UE_PER_ROUND * num_ues_unique_round

    total_flops_all_rounds += flops_this_round
    total_bytes_all_rounds += bytes_this_round
    total_unique_ues_updates += num_ues_unique_round
    total_ap_ue_instances += total_selected_instances

    _, acc_r = global_model.evaluate(test_ds, verbose=0)
    print(f"   ↳ UEs únicos: {num_ues_unique_round} | instâncias (AP,UE): {total_selected_instances} | cluster médio={avg_cluster:.2f}")
    print(f"   ↳ acc={acc_r*100:.2f}% | FLOPs: {pretty_num(flops_this_round)} | Bytes (↓↑): {bytes_this_round/1e6:.2f} MB")

loss, acc = global_model.evaluate(test_ds, verbose=0)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")

print("\n==================== RELATÓRIO FINAL ====================")
print(f"APs (M):                         {NUM_AP}")
print(f"UEs totais (K):                  {TOTAL_USERS}")
print(f"|D_j| (amostras por UE):          {SAMPLES_PER_USER}  (igual ao artigo)")
print(f"K por AP:                        {'ALL' if USE_ALL_USERS_PER_AP else K_PER_AP}")
print(f"USE_RL:                          {USE_RL} (RL_REWARD_MODE={RL_REWARD_MODE})")
print(f"Cluster combine:                 {CLUSTER_COMBINE_MODE}")
print(f"Parâmetros do modelo:            {n_params:,}")
print(f"FLOPs/amostra (forward):         {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):          {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)")
print(f"Total UEs únicos treinados:      {total_unique_ues_updates}")
print(f"Total instâncias (AP,UE):        {total_ap_ue_instances}")
print(f"FLOPs totais (treino local):     {pretty_num(total_flops_all_rounds)}FLOPs")
print(f"Bytes totais (↓↑, float32):      {total_bytes_all_rounds/1e6:.2f} MB")
print("=========================================================\n")

# fairness por "rounds que o UE participou"
jain_round = jain_index(ue_round_participations)

print("=========== FAIRNESS (UEs ÚNICOS POR RODADA) ===========")
print(f"Participações (rounds) mín / máx: {ue_round_participations.min():.0f} / {ue_round_participations.max():.0f}")
print(f"Média / desvio padrão:           {ue_round_participations.mean():.2f} / {ue_round_participations.std():.2f}")
print(f"Índice de Jain (rounds por UE):  {jain_round:.4f}")
print("========================================================\n")

# só para você enxergar o quanto repetiu em APs
jain_inst = jain_index(ue_ap_instances)
print("=========== ESTATÍSTICA (INSTÂNCIAS AP×UE) ============")
print(f"Instâncias (AP,UE) mín / máx:     {ue_ap_instances.min():.0f} / {ue_ap_instances.max():.0f}")
print(f"Média / desvio padrão:            {ue_ap_instances.mean():.2f} / {ue_ap_instances.std():.2f}")
print(f"Índice de Jain (instâncias):      {jain_inst:.4f}")
print("======================================================\n")
