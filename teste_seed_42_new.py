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
NUM_AP = 16                 # M = 16 (fixo)
K_USERS = 20                # K ∈ {20,40}
TOTAL_USERS = K_USERS

# Treino local (mantido como estava no seu código)
EPOCHS_LOCAL = 5
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Validação / reforço
VAL_SAMPLES = 5000
LAMBDA_OVERFIT = 0.15
TEMP = 0.9
WEIGHT_FLOOR, WEIGHT_CEIL = 0.8, 1.2

ALPHA_LEVEL = 0.7
BETA_GAIN  = 0.3

# Seleção top-K por AP
USE_ALL_USERS_PER_AP = False
K_PER_AP = 4

# Exploração/justiça
EPS = 0.05
TEMP_SEL = 0.8
FAIR_FRACTION = 0.25

# ============================================================
# ============== “CELL-FREE” SEM PRÉ-FILTRO ==================
# ============================================================
ap_users = {ap: list(range(TOTAL_USERS)) for ap in range(NUM_AP)}

PATHLOSS_EXP = 3.7
MIN_DIST = 0.02
SHADOWING_STD_DB = 6.0
LINK_QUALITY_GAMMA = 0.25  # peso do link no score

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

    beta = pl * shadow_lin  # (M,K)
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

# Normaliza beta por AP para virar link_quality[ap][ue] em [0,1]
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
#   CORRIGIDO: usa TODO o CIFAR-10 treino (50k) e divide
#   igualmente: |D_j| = |D|/K (igual ao artigo)
#   + balanceado por classe (IID “perfeito”)
# ============================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = y_train.squeeze().astype(np.int32)
y_test  = y_test.squeeze().astype(np.int32)

TOTAL_TRAIN = len(x_train)  # CIFAR-10 train = 50.000 (|D| do artigo)
if TOTAL_TRAIN % TOTAL_USERS != 0:
    raise ValueError(
        f"|D|={TOTAL_TRAIN} não é divisível por K={TOTAL_USERS}. "
        f"No artigo, K∈{{20,40}} divide exatamente 50.000."
    )

SAMPLES_PER_USER = TOTAL_TRAIN // TOTAL_USERS  # <-- IGUAL AO ARTIGO

x_val = x_test[:VAL_SAMPLES]
y_val = y_test[:VAL_SAMPLES]

def make_iid_splits_by_class_full(x, y, K, seed=42):
    """
    IID balanceado por classe usando TODO o treino:
    - CIFAR-10 tem 5000 amostras por classe
    - K=20 -> 250 por classe por usuário (2500 total)
    - K=40 -> 125 por classe por usuário (1250 total)
    """
    rng = np.random.default_rng(seed)
    num_classes = int(np.max(y)) + 1

    per_user_idx = [[] for _ in range(K)]
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        # round-robin garante mesma qtde por classe para todos
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
# ===================== MODELO (IGUAL AO ARTIGO) =============
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
        optimizer='adam',  # mantido como estava no seu loop
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# ===================== BANDIT (EMA) =========================
# ============================================================
class BanditSelector:
    def __init__(self, ap_to_users, ema_alpha=0.25):
        self.ema_alpha = ema_alpha
        self.n = defaultdict(lambda: defaultdict(int))
        self.q = defaultdict(lambda: defaultdict(float))
        self.last_round = defaultdict(lambda: defaultdict(int))
        self.ap_to_users = ap_to_users

    def mark_participation(self, ap, uid, round_idx):
        self.n[ap][uid] += 1
        self.last_round[ap][uid] = round_idx

    def update(self, ap, uid, reward):
        q_old = self.q[ap][uid]
        a = self.ema_alpha
        self.q[ap][uid] = (1 - a) * q_old + a * float(reward)

    def select_all(self, ap):
        return self.ap_to_users[ap]

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

def weights_from_q_softmax(ap, qdict, users,
                           floor=WEIGHT_FLOOR, ceil=WEIGHT_CEIL, temp=TEMP, eps=1e-9):
    qs = np.array([qdict[ap][u] for u in users], dtype=np.float32)

    if len(qs) >= 2:
        mu, sigma = float(np.mean(qs)), float(np.std(qs) + 1e-9)
        qs = np.clip(qs, mu - 2.0*sigma, mu + 2.0*sigma)

    qs = (qs - np.max(qs))
    sm = np.exp(qs / max(temp, 1e-3))
    sm = sm / (np.sum(sm) + eps)
    sm_scaled = sm * len(users)

    smin, smax = float(np.min(sm_scaled)), float(np.max(sm_scaled))
    if smax - smin < 1e-6:
        return {u: 1.0 for u in users}

    s01 = (sm_scaled - smin) / (smax - smin)
    mults = floor + (ceil - floor) * s01
    mults = mults / (float(np.mean(mults)) + eps)
    return {u: float(m) for u, m in zip(users, mults)}

# ============================================================
# ===================== SELEÇÃO TOP-K ========================
# ============================================================
def softmax_scores(vec, temp=1.0, eps=1e-9):
    v = np.array(vec, dtype=np.float32)
    v = v - np.max(v)
    sm = np.exp(v / max(temp, 1e-3))
    return sm / (np.sum(sm) + eps)

def select_topk_no_prefilter(ap, bandit, K, round_idx,
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

    if np.random.rand() < eps:
        return list(np.random.choice(users, size=K, replace=False))

    qs = np.array([bandit.q[ap][u] for u in users], dtype=np.float32)

    ages = np.array(
        [round_idx - bandit.last_round[ap][u] if bandit.last_round[ap][u] else round_idx + 1
         for u in users], dtype=np.float32
    )
    if float(np.max(ages)) > 0:
        ages = ages / (float(np.max(ages)) + 1e-9)

    lq = np.array([link_q_dict.get(u, 0.0) for u in users], dtype=np.float32)

    fairness_boost = 0.10
    scores = qs + fairness_boost * ages + gamma_link * lq

    k_fair = max(1, int(np.ceil(fair_fraction * K)))
    tie = np.array([bandit.n[ap][u] for u in users], dtype=np.int32)
    rank_fair = np.lexsort((tie, -ages))
    fair_candidates = [users[i] for i in rank_fair[:k_fair]]

    remaining = K - len(fair_candidates)
    if remaining > 0:
        idx_map = {u:i for i,u in enumerate(users)}
        mask = np.ones(len(users), dtype=bool)
        for u in fair_candidates:
            mask[idx_map[u]] = False

        users_left = [u for u, m in zip(users, mask) if m]
        probs_left = softmax_scores([scores[idx_map[u]] for u in users_left], temp=temp)
        chosen = list(np.random.choice(users_left, size=remaining, replace=False, p=probs_left))
        return fair_candidates + chosen

    return fair_candidates[:K]

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
bandit = BanditSelector(ap_users, ema_alpha=0.25)
global_model = build_model(num_classes=10)

n_params = count_params(global_model)
infer_flops = inference_flops_tf(global_model)
train_flops_per_sample = train_flops_per_sample_from_infer(infer_flops)

BYTES_PER_PARAM = 4
BYTES_PER_UE_PER_ROUND = 2 * n_params * BYTES_PER_PARAM

print("\n===== CONFIG =====")
print(f"M (APs) = {NUM_AP}")
print(f"K (UEs) = {TOTAL_USERS}")
print(f"|D_j| (amostras por UE) = {SAMPLES_PER_USER}  (igual ao artigo)\n")

print("===== MODELO =====")
print(f"Parâmetros: {n_params:,}")
print(f"FLOPs/amostra (forward): {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):  {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)\n")

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

ue_participations = np.zeros(TOTAL_USERS, dtype=np.int32)

total_flops_all_rounds = 0.0
total_bytes_all_rounds = 0.0
total_selected_updates = 0

ALL_UES = list(range(TOTAL_USERS))

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    _, acc_val_global = global_model.evaluate(val_ds, verbose=0)

    selected_by_ap = {}
    for ap in range(NUM_AP):
        if USE_ALL_USERS_PER_AP:
            selected_by_ap[ap] = ALL_UES[:]
        else:
            selected_by_ap[ap] = select_topk_no_prefilter(
                ap=ap,
                bandit=bandit,
                K=K_PER_AP,
                round_idx=r,
                candidates=ALL_UES,
                link_q_dict=link_quality[ap],
                temp=TEMP_SEL,
                eps=EPS,
                fair_fraction=FAIR_FRACTION,
                gamma_link=LINK_QUALITY_GAMMA
            )

    selected_union = sorted(set(u for ap in range(NUM_AP) for u in selected_by_ap[ap]))
    num_ues_unique_round = len(selected_union)

    local_cache = {ap: {} for ap in range(NUM_AP)}
    total_selected_instances = 0

    for ap in range(NUM_AP):
        for uid in selected_by_ap[ap]:
            total_selected_instances += 1

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

            _, acc_local = local_model.evaluate(x_u, y_u, verbose=0)
            _, acc_val_u = local_model.evaluate(val_ds, verbose=0)

            reward_level = float(acc_val_u)
            reward_gain  = float(acc_val_u - acc_val_global)
            reward = (
                ALPHA_LEVEL * reward_level
                + BETA_GAIN * reward_gain
                - LAMBDA_OVERFIT * max(0.0, float(acc_local - acc_val_u))
            )

            local_cache[ap][uid] = (local_model.get_weights(), len(x_u), reward)
            ue_participations[uid] += 1

    for ap in range(NUM_AP):
        for uid in selected_by_ap[ap]:
            bandit.mark_participation(ap, uid, r)
            bandit.update(ap, uid, local_cache[ap][uid][2])

    ap_models_weights = []
    ap_sizes = []
    for ap in range(NUM_AP):
        uids = selected_by_ap[ap]
        if len(uids) == 0:
            ap_models_weights.append(global_model.get_weights())
            ap_sizes.append(0.0)
            continue

        weights_list = [local_cache[ap][uid][0] for uid in uids]
        sizes = [local_cache[ap][uid][1] for uid in uids]

        mults = weights_from_q_softmax(ap, bandit.q, users=uids)
        sizes_weighted = [n * mults[uid] for n, uid in zip(sizes, uids)]

        ap_w = fedavg(weights_list, sizes_weighted)
        ap_models_weights.append(ap_w)
        ap_sizes.append(float(np.sum(sizes_weighted)))

    new_global_weights = fedavg(ap_models_weights, ap_sizes)
    if new_global_weights is not None:
        global_model.set_weights(new_global_weights)

    flops_this_round = train_flops_per_sample * SAMPLES_PER_USER * EPOCHS_LOCAL * total_selected_instances
    bytes_this_round = BYTES_PER_UE_PER_ROUND * total_selected_instances

    total_flops_all_rounds += flops_this_round
    total_bytes_all_rounds += bytes_this_round
    total_selected_updates += total_selected_instances

    _, acc_r = global_model.evaluate(test_ds, verbose=0)
    print(f"   ↳ UEs únicos: {num_ues_unique_round} | treinos (AP,UE): {total_selected_instances} | acc={acc_r*100:.2f}%")
    print(f"   ↳ FLOPs: {pretty_num(flops_this_round)} | Bytes (↓↑): {bytes_this_round/1e6:.2f} MB")

loss, acc = global_model.evaluate(test_ds, verbose=0)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")

print("\n==================== RELATÓRIO FINAL ====================")
print(f"APs (M):                         {NUM_AP}")
print(f"UEs totais (K):                  {TOTAL_USERS}  (K ∈ {{20,40}})")
print(f"|D_j| (amostras por UE):          {SAMPLES_PER_USER}  (igual ao artigo)")
print(f"K selecionados por AP:           {'ALL' if USE_ALL_USERS_PER_AP else K_PER_AP}")
print(f"Parâmetros do modelo:            {n_params:,}")
print(f"FLOPs/amostra (forward):         {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):          {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)")
print(f"Total de treinos (AP,UE) somados:{total_selected_updates}")
print(f"FLOPs totais (treino local):     {pretty_num(total_flops_all_rounds)}FLOPs")
print(f"Bytes totais (↓↑, float32):      {total_bytes_all_rounds/1e6:.2f} MB")
print("=========================================================\n")

jain_part = jain_index(ue_participations)
print("==================== FAIRNESS DE PARTICIPAÇÃO (POR UE) ====================")
print(f"Participações (treinos) mín / máx:     {ue_participations.min():.0f} / {ue_participations.max():.0f}")
print(f"Média / desvio padrão:                {ue_participations.mean():.2f} / {ue_participations.std():.2f}")
print(f"Índice de Jain (treinos por UE):       {jain_part:.4f}")
print("============================================================================\n")
