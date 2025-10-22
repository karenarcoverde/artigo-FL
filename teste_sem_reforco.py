import os, random, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from collections import defaultdict

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
NUM_AP = 4
USERS_PER_AP = 10
TOTAL_USERS = NUM_AP * USERS_PER_AP

# Treino local
SAMPLES_PER_USER = 300
EPOCHS_LOCAL = 5
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Validação (para avaliação global/relatórios; sem RL)
VAL_SAMPLES = 5000

# -------------------------
# Seleção SEM reforço
# -------------------------
USE_ALL_USERS_PER_AP = True    # ← usa todos os UEs por AP em todas as rodadas
K_PER_AP = 4                   # (não usado quando USE_ALL_USERS_PER_AP=True)

# Justiça/rotatividade básica (sem Q) — mantém para compatibilidade
EPS = 0.05
FAIR_FRACTION = 0.25

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

# Val set (para métricas globais)
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
# Seleção (sem RL) com justiça simples por recência (compatível com "usar todos")
# -------------------------
class SimpleSelector:
    def __init__(self, ap_to_users):
        self.ap_to_users = ap_to_users
        self.n = defaultdict(lambda: defaultdict(int))
        self.last_round = defaultdict(lambda: defaultdict(int))

    def mark_participation(self, ap, uid, round_idx):
        self.n[ap][uid] += 1
        self.last_round[ap][uid] = round_idx

    def select_all(self, ap):
        return self.ap_to_users[ap]

    def select_topk_fair(self, ap, K, round_idx, eps=EPS, fair_fraction=FAIR_FRACTION):
        users = self.ap_to_users[ap]
        K = min(K, len(users))
        if K == len(users):
            return users[:]
        if np.random.rand() < eps:
            return list(np.random.choice(users, size=K, replace=False))
        ages = np.array(
            [round_idx - self.last_round[ap][u] if self.last_round[ap][u] else round_idx+1
             for u in users], dtype=np.int64
        )
        tie = np.array([self.n[ap][u] for u in users], dtype=np.int64)
        rank_fair = np.lexsort((tie, -ages))  # desc idade, asc n
        k_fair = max(1, int(math.ceil(fair_fraction * K)))
        fair_candidates = [users[i] for i in rank_fair[:k_fair]]
        remaining = K - len(fair_candidates)
        if remaining > 0:
            rest = [u for u in users if u not in fair_candidates]
            chosen = list(np.random.choice(rest, size=remaining, replace=False))
            selected = fair_candidates + chosen
        else:
            selected = fair_candidates[:K]
        return selected

# -------------------------
# FedAvg (ponderado pelo tamanho do dataset local)
# -------------------------
def fedavg(weights_list, sizes):
    N = float(np.sum(sizes))
    new_weights = []
    for layer_ws in zip(*weights_list):
        stack = np.stack(layer_ws, axis=0)
        coeffs = (np.array(sizes)/N).reshape((-1,) + (1,)*(stack.ndim-1))
        new_weights.append(np.sum(coeffs * stack, axis=0))
    return new_weights

# ============================================================
# =======  MÉTRICAS DE FLOPs / PARÂMETROS / TRÁFEGO  ========
# ============================================================
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

def inference_flops_tf(model, input_shape=(1,32,32,3), dtype=tf.float32):
    """FLOPs do forward por amostra via TF Profiler (grafo congelado)."""
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
        # continua reduzindo
    return f"{x:.2f}E"

# -------------------------
# Loop Federado Hierárquico (SEM RL, usando TODOS os UEs)
# -------------------------
selector = SimpleSelector(ap_users)
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
total_selected_updates = 0

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    # Avaliação global no val (referência)
    _ = global_model.evaluate(val_ds, verbose=0)

    ap_models_weights = []
    ap_sizes = []
    selected_this_round = 0

    for ap in range(NUM_AP):
        if USE_ALL_USERS_PER_AP:
            selected_user_ids = selector.select_all(ap)   # ← todos os 10 por AP
        else:
            selected_user_ids = selector.select_topk_fair(ap, K_PER_AP, round_idx=r,
                                                         eps=EPS, fair_fraction=FAIR_FRACTION)
        selected_this_round += len(selected_user_ids)

        # print opcional para checagem:
        # print(f"AP {ap}: {len(selected_user_ids)} UEs -> {selected_user_ids}")

        local_weights_list = []
        local_sizes = []

        for uid in selected_user_ids:
            selector.mark_participation(ap, uid, r)

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

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # = SAMPLES_PER_USER

        # Agregação no AP (FedAvg puro)
        ap_weights = fedavg(local_weights_list, local_sizes)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes))

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
    sel_info = f"todos ({USERS_PER_AP})"
    print(f"   ↳ APs usando {sel_info} por AP | UEs nesta rodada: {selected_this_round} | acc(test)={acc_r*100:.2f}%")
    print(f"   ↳ FLOPs desta rodada: {pretty_num(flops_this_round)} | Bytes (≈float32): {bytes_this_round/1e6:.2f} MB")

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(test_ds, verbose=2)
print(f"\n🎯 Acurácia final do modelo global (todos UEs): {acc * 100:.2f}%")

# -------------------------
# RELATÓRIO FINAL DE COMPLEXIDADE
# -------------------------
print("\n==================== RELATÓRIO FINAL ====================")
print(f"Parâmetros do modelo:           {n_params:,}")
print(f"FLOPs/amostra (forward):        {pretty_num(infer_flops)}FLOPs")
print(f"FLOPs/amostra (treino):         {pretty_num(train_flops_per_sample)}FLOPs (≈2.5×)")
print(f"Total de atualizações (UEs):     {total_selected_updates} (somatório de UEs selecionados em todas as rodadas)")
print(f"FLOPs totais (treino local):    {pretty_num(total_flops_all_rounds)}FLOPs")
print(f"Bytes totais (↓↑, float32):     {total_bytes_all_rounds/1e6:.2f} MB")
print(  "=========================================================\n")
