import os, random, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from collections import defaultdict

# -------------------------
# Índice de Jain (fairness)
# -------------------------
def jain_index(x):
    """
    Índice de Jain:
        J(x) = ( (sum_i x_i)^2 ) / ( n * sum_i x_i^2 )
    x: array-like com valores não-negativos (ex.: nº de participações por UE)
    Retorna valor em [0, 1]; quanto mais perto de 1, mais justo.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0
    s = x.sum()
    s2 = np.sum(x**2)
    if s2 == 0:
        return 0.0
    return (s**2) / (n * s2)


# -------------------------
# Sementes / determinismo
# -------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# -------------------------
# Configurações
# -------------------------
NUM_AP = 5
USERS_PER_AP = 20
TOTAL_USERS = NUM_AP * USERS_PER_AP

# Treino local
SAMPLES_PER_USER = 300        # ex.: 300 (antes era 100)
EPOCHS_LOCAL = 5              # se reduzir K_PER_AP, pode subir p/ 6-7
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Validação / reforço
VAL_SAMPLES = 5000            # val maior p/ recompensa
LAMBDA_OVERFIT = 0.15         # penalização leve do overfit local
TEMP = 0.9                    # temperatura p/ ponderação suave (agregação)
WEIGHT_FLOOR, WEIGHT_CEIL = 0.8, 1.2  # multiplicadores 0.8–1.2

# Blend da recompensa
ALPHA_LEVEL = 0.7             # peso do nível (acc_val)
BETA_GAIN  = 0.3              # peso do ganho marginal

# -------------------------
# Seleção COM reforço (top-K + justiça)
# -------------------------
USE_ALL_USERS_PER_AP = True  # << agora selecionamos menos UEs por AP
K_PER_AP = 2                  # ex.: de 10 -> 4 por AP

# Exploração/justiça
EPS = 0.05                    # ε-greedy (exploração)
TEMP_SEL = 0.8                # temperatura softmax na seleção (0.6–1.2)
FAIR_FRACTION = 0.25          # fração dos K reservada p/ recência
PARTICIPATION_WINDOW = 5      # (não obrigatório aqui; mantido p/ extensão)

# -------------------------
# Dados (IID com mais amostras por UE)
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

# Amostra um pool estratificado com TOTAL_USERS * SAMPLES_PER_USER
TOTAL_TRAIN = TOTAL_USERS * SAMPLES_PER_USER
x_pool, _, y_pool, _ = train_test_split(
    x_train, y_train,
    train_size=TOTAL_TRAIN,
    stratify=y_train,
    random_state=SEED,
    shuffle=True
)

# Embaralha e fatia igualmente (IID) em blocos de SAMPLES_PER_USER
rng = np.random.default_rng(SEED)
perm = rng.permutation(TOTAL_TRAIN)
x_pool = x_pool[perm]; y_pool = y_pool[perm]

user_data = []
for u in range(TOTAL_USERS):
    s = u * SAMPLES_PER_USER
    e = (u + 1) * SAMPLES_PER_USER
    user_data.append((x_pool[s:e], y_pool[s:e]))

# Val set compartilhado (maior)
x_val = x_test[:VAL_SAMPLES]
y_val = y_test[:VAL_SAMPLES]

# Índices de usuários por AP
ap_users = {
    ap: list(range(ap*USERS_PER_AP, (ap+1)*USERS_PER_AP))
    for ap in range(NUM_AP)
}

# -------------------------
# Modelo base
# -------------------------
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Bandit (EMA + rastreio de recência)
# -------------------------
class BanditSelector:
    def __init__(self, ap_to_users, ema_alpha=0.25):
        self.ema_alpha = ema_alpha
        self.n = defaultdict(lambda: defaultdict(int))          # contagem de participações
        self.q = defaultdict(lambda: defaultdict(float))        # valor/aprendizado
        self.last_round = defaultdict(lambda: defaultdict(int)) # última rodada que participou
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

# -------------------------
# FedAvg (ponderado)
# -------------------------
def fedavg(weights_list, sizes):
    N = float(np.sum(sizes))
    new_weights = []
    for layer_ws in zip(*weights_list):
        stack = np.stack(layer_ws, axis=0)
        coeffs = (np.array(sizes)/N).reshape((-1,) + (1,)*(stack.ndim-1))
        new_weights.append(np.sum(coeffs * stack, axis=0))
    return new_weights

# Pesos SUAVES via softmax com temperatura e normalização p/ média≈1
def weights_from_q_softmax(ap, qdict, users=None, floor=WEIGHT_FLOOR, ceil=WEIGHT_CEIL, temp=TEMP, eps=1e-9):
    if users is None:
        users = ap_users[ap]
    qs = np.array([qdict[ap][u] for u in users], dtype=np.float32)

    # z-score clipping (evita outliers derrubarem as escolhas)
    if len(qs) >= 2:
        mu, sigma = float(np.mean(qs)), float(np.std(qs) + 1e-9)
        qs = np.clip(qs, mu - 2.0*sigma, mu + 2.0*sigma)

    qs = (qs - np.max(qs))
    sm = np.exp(qs / max(temp, 1e-3))
    sm = sm / (np.sum(sm) + eps)              # soma=1
    sm_scaled = sm * len(users)               # média ≈1

    smin, smax = float(np.min(sm_scaled)), float(np.max(sm_scaled))
    if smax - smin < 1e-6:
        return {u: 1.0 for u in users}
    s01 = (sm_scaled - smin) / (smax - smin)
    mults = floor + (ceil - floor) * s01
    mults = mults / (float(np.mean(mults)) + eps)  # média 1
    return {u: float(m) for u, m in zip(users, mults)}

# -------------------------
# Seleção top-K com justiça/recência
# -------------------------
def softmax_scores(vec, temp=1.0, eps=1e-9):
    v = np.array(vec, dtype=np.float32)
    v = v - np.max(v)
    sm = np.exp(v / max(temp, 1e-3))
    return sm / (np.sum(sm) + eps)

def select_topk_fair(ap, bandit, K, round_idx,
                     temp=TEMP_SEL, eps=EPS, fair_fraction=FAIR_FRACTION, window=PARTICIPATION_WINDOW):
    users = ap_users[ap]
    K = min(K, len(users))
    if K == len(users):
        return users[:]

    # 1) ε-greedy: exploração total ocasional
    if np.random.rand() < eps:
        return list(np.random.choice(users, size=K, replace=False))

    # 2) Score = Q + pequeno bônus de recência
    qs = np.array([bandit.q[ap][u] for u in users], dtype=np.float32)
    ages = np.array(
        [round_idx - bandit.last_round[ap][u] if bandit.last_round[ap][u] else round_idx+1
         for u in users], dtype=np.float32
    )
    if np.max(ages) > 0:
        ages = ages / (np.max(ages) + 1e-9)
    fairness_boost = 0.1
    scores = qs + fairness_boost * ages

    # 3) Reserva fração FAIR_K por recência
    k_fair = max(1, int(np.ceil(fair_fraction * K)))
    tie = np.array([bandit.n[ap][u] for u in users], dtype=np.int32)
    # ordenar por "mais tempo sem participar" (idade alta), desempate por menor n
    rank_fair = np.lexsort((tie, -ages))
    fair_candidates = [users[i] for i in rank_fair[:k_fair]]

    # 4) Completa com amostragem enviesada por score (softmax)
    remaining = K - len(fair_candidates)
    if remaining > 0:
        mask = np.ones(len(users), dtype=bool)
        for u in fair_candidates:
            mask[users.index(u)] = False
        users_left = [u for u, m in zip(users, mask) if m]
        probs_left = softmax_scores([bandit.q[ap][u] for u in users_left], temp=temp)
        chosen = list(np.random.choice(users_left, size=remaining, replace=False, p=probs_left))
        selected = fair_candidates + chosen
    else:
        selected = fair_candidates[:K]

    return selected

# ============================================================
# =======  MÉTRICAS DE FLOPs / PARÂMETROS / TRÁFEGO  ========
# ============================================================
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

def inference_flops_tf(model, input_shape=(1,32,32,3), dtype=tf.float32):
    """
    FLOPs do forward por amostra via TF Profiler (grafo congelado).
    """
    @tf.function
    def forward(x):
        return model(x, training=False)

    concrete = forward.get_concrete_function(tf.TensorSpec(input_shape, dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops  # FLOPs forward / amostra

def train_flops_per_sample_from_infer(infer_flops):
    # regra prática: treino ≈ 2.5 × forward (forward + backward + update)
    return infer_flops * 2.5

def pretty_num(x):
    for u in ["", "k", "M", "G", "T", "P"]:
        if abs(x) < 1000: return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{x:.2f}E"

# -------------------------
# Loop Federado Hierárquico
# -------------------------
bandit = BanditSelector(ap_users, ema_alpha=0.25)
global_model = build_model()

# ---- MÉTRICAS FIXAS DO MODELO (1x ao início) ----
n_params = count_params(global_model)
infer_flops = inference_flops_tf(global_model)               # FLOPs fwd por amostra
train_flops_per_sample = train_flops_per_sample_from_infer(infer_flops)

# Para bytes estimados (float32)
BYTES_PER_PARAM = 4
BYTES_PER_UE_PER_ROUND = 2 * n_params * BYTES_PER_PARAM     # download + upload

print("\n===== MODELO =====")
print(f"Parâmetros: {n_params:,}")
print(f"FLOPs/amostra (forward): {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):  {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)\n")

# Acumuladores globais
total_flops_all_rounds = 0.0
total_bytes_all_rounds = 0.0
total_selected_updates = 0     # soma dos UEs selecionados ao longo das rodadas

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    # ACC global no val — base p/ "ganho"
    _, acc_val_global = global_model.evaluate(val_ds, verbose=0)

    ap_models_weights = []
    ap_sizes = []
    selected_this_round = 0  # zera a cada rodada

    for ap in range(NUM_AP):
        if USE_ALL_USERS_PER_AP:
            selected_user_ids = bandit.select_all(ap)
        else:
            selected_user_ids = select_topk_fair(
                ap, bandit, K_PER_AP, round_idx=r,
                temp=TEMP_SEL, eps=EPS, fair_fraction=FAIR_FRACTION, window=PARTICIPATION_WINDOW
            )
        selected_this_round += len(selected_user_ids)

        local_weights_list = []
        local_sizes = []

        for uid in selected_user_ids:
            bandit.mark_participation(ap, uid, r)  # registra recência/participação

            # cada UE começa do global
            local_model = build_model()
            local_model.set_weights(global_model.get_weights())

            x_u, y_u = user_data[uid]
            local_model.fit(
                x_u, y_u,
                epochs=EPOCHS_LOCAL,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True
            )

            # Recompensa: nível + ganho marginal - overfit local
            _, acc_local = local_model.evaluate(x_u,   y_u,   verbose=0)
            _, acc_val_u = local_model.evaluate(val_ds,        verbose=0)
            reward_level = acc_val_u
            reward_gain  = acc_val_u - acc_val_global
            reward = ALPHA_LEVEL * reward_level + BETA_GAIN * reward_gain \
                     - LAMBDA_OVERFIT * max(0.0, acc_local - acc_val_u)

            bandit.update(ap, uid, reward)

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # = SAMPLES_PER_USER

        # Pondera SUAVEMENTE apenas os selecionados
        mults = weights_from_q_softmax(ap, bandit.q, users=selected_user_ids)
        local_sizes_weighted = [n * mults[uid] for n, uid in zip(local_sizes, selected_user_ids)]

        # Agregação no AP
        ap_weights = fedavg(local_weights_list, local_sizes_weighted)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes_weighted))

    # ---- FLOPs e Bytes desta rodada ----
    flops_this_round = train_flops_per_sample * SAMPLES_PER_USER * EPOCHS_LOCAL * selected_this_round
    bytes_this_round = BYTES_PER_UE_PER_ROUND * selected_this_round

    total_flops_all_rounds += flops_this_round
    total_bytes_all_rounds += bytes_this_round
    total_selected_updates += selected_this_round

    # Agregação global (FedAvg)
    new_global_weights = fedavg(ap_models_weights, ap_sizes)
    global_model.set_weights(new_global_weights)

    # Acompanhar evolução
    loss_r, acc_r = global_model.evaluate(test_ds, verbose=0)
    if USE_ALL_USERS_PER_AP:
        sel_info = f"todos ({USERS_PER_AP})"
    else:
        sel_info = f"top-{K_PER_AP}"
    print(f"   ↳ APs usando {sel_info} por AP | UEs nesta rodada: {selected_this_round} | acc={acc_r*100:.2f}%")
    print(f"   ↳ FLOPs desta rodada: {pretty_num(flops_this_round)} | Bytes (≈float32) desta rodada: {bytes_this_round/1e6:.2f} MB")

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(test_ds, verbose=2)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")

# -------------------------
# RELATÓRIO FINAL DE COMPLEXIDADE
# -------------------------
print("\n==================== RELATÓRIO FINAL ====================")
print(f"Parâmetros do modelo:           {n_params:,}")
print(f"FLOPs/amostra (forward):        {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):         {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)")
print(f"Total de atualizações (UEs):    {total_selected_updates} (somatório de UEs selecionados em todas as rodadas)")
print(f"FLOPs totais (treino local):    {pretty_num(total_flops_all_rounds)}FLOPs")
print(f"Bytes totais (↓↑, float32):     {total_bytes_all_rounds/1e6:.2f} MB")
print("=========================================================\n")

# -------------------------
# FAIRNESS: ÍNDICE DE JAIN DAS PARTICIPAÇÕES
# -------------------------
participations = []
for ap in range(NUM_AP):
    for uid in ap_users[ap]:
        participations.append(bandit.n[ap][uid])

participations = np.array(participations, dtype=float)
jain_part = jain_index(participations)

print("\n==================== FAIRNESS DE PARTICIPAÇÃO ====================")
print(f"Número total de UEs:                  {len(participations)}")
print(f"Participações mín / máx:              {participations.min():.0f} / {participations.max():.0f}")
print(f"Média / desvio padrão das particip.:  {participations.mean():.2f} / {participations.std():.2f}")
print(f"Índice de Jain das participações:     {jain_part:.4f}")
print("==============================================================\n")