import os, random, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

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

SAMPLES_PER_USER = 300
EPOCHS_LOCAL_BASE = 5         # épocas por UE quando todos participam
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Non-IID por Dirichlet
USE_DIRICHLET = True
ALPHA_DIR = 0.5               # ↓ mais não-IID; ↑ mais próximo de IID

# Seleção parcial com bandit (UCB1)
USE_PARTIAL = True
K_PER_AP = 5                  # 4–6 é um bom ponto
WARM_ROUNDS = 5               # primeiras rodadas: todos os UEs

# Val e recompensa
VAL_SAMPLES = 5000
ALPHA_LEVEL = 0.7             # peso do nível de acc_val
BETA_GAIN  = 0.3              # peso do ganho marginal (acc_val - acc_val_global)
LAMBDA_OVERFIT = 0.15         # penaliza (acc_local - acc_val)+

# -------------------------
# Modelo base (igual ao baseline)
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
# FedAvg (clássico)
# -------------------------
def fedavg(weights_list, sizes):
    N = float(np.sum(sizes))
    new_weights = []
    for layer_ws in zip(*weights_list):
        stack = np.stack(layer_ws, axis=0)
        coeffs = (np.array(sizes)/N).reshape((-1,) + (1,)*(stack.ndim-1))
        new_weights.append(np.sum(coeffs * stack, axis=0))
    return new_weights

# -------------------------
# Particionamento non-IID por Dirichlet
# -------------------------
def dirichlet_partition(x, y, num_users, samples_per_user, alpha, seed=SEED, num_classes=10):
    rng = np.random.default_rng(seed)

    # índices por classe
    class_indices = {c: np.where(y.flatten() == c)[0] for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # p_u ~ Dir(alpha)
    P = rng.dirichlet(alpha * np.ones(num_classes), size=num_users)
    # contagens Multinomial(S, p_u)
    counts = np.zeros((num_users, num_classes), dtype=int)
    for u in range(num_users):
        counts[u] = rng.multinomial(samples_per_user, P[u])

    # limitar à disponibilidade por classe (ajuste proporcional)
    for c in range(num_classes):
        need = int(counts[:, c].sum())
        have = int(len(class_indices[c]))
        if need <= have:
            continue
        scale = have / max(1, need)
        scaled = np.floor(counts[:, c] * scale).astype(int)
        rest = have - int(scaled.sum())
        residuals = (counts[:, c] * scale) - scaled
        order = np.argsort(-residuals)
        for i in range(rest):
            scaled[order[i]] += 1
        counts[:, c] = scaled

    # atribui índices concretos
    user_indices = [[] for _ in range(num_users)]
    ptr = {c: 0 for c in range(num_classes)}
    for u in range(num_users):
        for c in range(num_classes):
            k = int(counts[u, c])
            if k <= 0: 
                continue
            start = ptr[c]; end = start + k
            idxs = class_indices[c][start:end]
            user_indices[u].extend(idxs.tolist())
            ptr[c] = end
        rng.shuffle(user_indices[u])

    # datasets por usuário
    user_data = []
    for u in range(num_users):
        idxs = np.array(user_indices[u], dtype=int)
        user_data.append((x[idxs], y[idxs]))
    return user_data

# -------------------------
# Thompson/UCB selector (UCB1 simples)
# -------------------------
class UCBSelector:
    """
    Um bandit por AP. Guarda média incremental e contagens n[ap][uid].
    select_k: escolhe K maiores por UCB = mean + c*sqrt(log(T)/n)
    """
    def __init__(self, ap_to_users, c=1.5):
        from collections import defaultdict
        self.c = c
        self.ap_users = ap_to_users
        self.t = defaultdict(int)                        # passos por AP
        self.n = defaultdict(lambda: defaultdict(int))   # contagens
        self.mean = defaultdict(lambda: defaultdict(float))  # média recompensa

    def select_k(self, ap, k):
        users = self.ap_users[ap]
        # warm-up interno: garantir 1 pull para cada UE
        cold = [u for u in users if self.n[ap][u] == 0]
        chosen = []
        if cold:
            chosen.extend(cold[:k])
        if len(chosen) >= k:
            return chosen[:k]

        # UCB
        self.t[ap] += 1
        T = max(1, self.t[ap])
        scores = []
        for u in users:
            if self.n[ap][u] == 0:
                score = float('inf')
            else:
                score = self.mean[ap][u] + self.c * math.sqrt(math.log(T) / self.n[ap][u])
            scores.append((score, u))
        scores.sort(reverse=True)
        for _, u in scores:
            if u not in chosen:
                chosen.append(u)
            if len(chosen) == k:
                break
        return chosen

    def update(self, ap, uid, reward):
        n = self.n[ap][uid]
        m = self.mean[ap][uid]
        n_new = n + 1
        m_new = m + (float(reward) - m) / n_new
        self.n[ap][uid] = n_new
        self.mean[ap][uid] = m_new

# -------------------------
# Dados
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
num_classes = 10

TOTAL_USERS = NUM_AP * USERS_PER_AP

if USE_DIRICHLET:
    # usa TODO o x_train; cada UE pega SAMPLES_PER_USER exemplos do conjunto total
    user_data = dirichlet_partition(
        x_train, y_train,
        num_users=TOTAL_USERS,
        samples_per_user=SAMPLES_PER_USER,
        alpha=ALPHA_DIR,
        seed=SEED,
        num_classes=num_classes
    )
else:
    # IID com estratificação e fatiamento igual
    TOTAL_TRAIN = TOTAL_USERS * SAMPLES_PER_USER
    x_pool, _, y_pool, _ = train_test_split(
        x_train, y_train,
        train_size=TOTAL_TRAIN,
        stratify=y_train,
        random_state=SEED,
        shuffle=True
    )
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(TOTAL_TRAIN)
    x_pool = x_pool[perm]; y_pool = y_pool[perm]
    user_data = []
    for u in range(TOTAL_USERS):
        s = u * SAMPLES_PER_USER
        e = (u + 1) * SAMPLES_PER_USER
        user_data.append((x_pool[s:e], y_pool[s:e]))

# Val set (só para recompensa/monitoramento)
VAL_SAMPLES = min(5000, len(x_test))
x_val = x_test[:VAL_SAMPLES]; y_val = y_test[:VAL_SAMPLES]

# Índices de usuários por AP
ap_users = { ap: list(range(ap*USERS_PER_AP, (ap+1)*USERS_PER_AP)) for ap in range(NUM_AP) }

# -------------------------
# Seleção e orçamento
# -------------------------
rng = np.random.default_rng(SEED)
selector = UCBSelector(ap_users, c=1.5)

def scaled_local_epochs(k, base_epochs=EPOCHS_LOCAL_BASE, total=USERS_PER_AP):
    scale = total / max(1, k)
    return max(5, int(np.ceil(base_epochs * scale)))

# -------------------------
# Loop Federado Hierárquico
# -------------------------
global_model = build_model()
val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    # Base para ganho marginal
    _, acc_val_global = global_model.evaluate(val_ds, verbose=0)

    ap_models_weights = []
    ap_sizes = []

    # orçamento: se parcial e já passou warm-up, aumenta épocas locais
    if USE_PARTIAL and r >= WARM_ROUNDS:
        E_LOCAL = scaled_local_epochs(K_PER_AP)
    else:
        E_LOCAL = EPOCHS_LOCAL_BASE

    for ap in range(NUM_AP):
        if (not USE_PARTIAL) or (r < WARM_ROUNDS):
            selected_user_ids = ap_users[ap]             # warm-up: todos
        else:
            selected_user_ids = selector.select_k(ap, min(K_PER_AP, len(ap_users[ap])))

        local_weights_list = []
        local_sizes = []

        current_global_w = global_model.get_weights()

        for uid in selected_user_ids:
            # cada UE começa do global
            local_model = build_model()
            local_model.set_weights(current_global_w)

            x_u, y_u = user_data[uid]
            local_model.fit(
                x_u, y_u,
                epochs=E_LOCAL,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True
            )

            # recompensa: nível + ganho marginal - penalização de overfit
            _, acc_local = local_model.evaluate(x_u,   y_u,   verbose=0)
            _, acc_val_u = local_model.evaluate(val_ds,        verbose=0)
            reward_level = float(acc_val_u)
            reward_gain  = float(acc_val_u - acc_val_global)
            reward = ALPHA_LEVEL*reward_level + BETA_GAIN*reward_gain \
                     - LAMBDA_OVERFIT*max(0.0, acc_local - acc_val_u)

            selector.update(ap, uid, reward)

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # SAMPLES_PER_USER

        # Agregação no AP (FedAvg padrão por tamanho)
        ap_weights = fedavg(local_weights_list, local_sizes)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes))

    # Agregação global (FedAvg)
    new_global_weights = fedavg(ap_models_weights, ap_sizes)
    global_model.set_weights(new_global_weights)

    # Acompanhar evolução
    _, acc_r = global_model.evaluate(test_ds, verbose=0)
    print(f"   ↳ após agregação dos APs: acc={acc_r*100:.2f}%")

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(test_ds, verbose=2)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")
