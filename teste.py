import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import math
from collections import defaultdict

# -------------------------
# Sementes / determinismo
# -------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
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
EPOCHS_LOCAL = 5
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Bandit-weighted FedAvg (estável)
VAL_SAMPLES = 5000            # val mais estável p/ recompensa
LAMBDA_OVERFIT = 0.15         # penalização leve do overfit local
TEMP = 0.9                    # temperatura mais alta -> pesos mais suaves
WEIGHT_FLOOR, WEIGHT_CEIL = 0.8, 1.2  # faixa de multiplicadores SUAVE

# Blend da recompensa
ALPHA_LEVEL = 0.7             # peso do nível (acc_val)
BETA_GAIN  = 0.3              # peso do ganho marginal

# -------------------------
# Dados
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

x_train, _, y_train, _ = train_test_split(
    x_train, y_train,
    train_size=TOTAL_USERS * 100,  # 100 amostras por usuário
    stratify=y_train,
    random_state=SEED,
    shuffle=True
)

# Val set compartilhado (maior)
x_val = x_test[:VAL_SAMPLES]
y_val = y_test[:VAL_SAMPLES]

# Fatias por usuário
user_data = []
for i in range(TOTAL_USERS):
    s, e = i*100, (i+1)*100
    user_data.append((x_train[s:e], y_train[s:e]))

# Índices de usuários por AP
ap_users = {
    ap: list(range(ap*USERS_PER_AP, (ap+1)*USERS_PER_AP))
    for ap in range(NUM_AP)
}

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
# Bandit (EMA nos valores)
# -------------------------
class BanditSelector:
    def __init__(self, ap_to_users, ema_alpha=0.25):
        self.ema_alpha = ema_alpha
        self.n = defaultdict(lambda: defaultdict(int))
        self.q = defaultdict(lambda: defaultdict(float))
        self.ap_to_users = ap_to_users
    def select_all(self, ap):
        return self.ap_to_users[ap]  # usamos TODOS os UEs (não perde orçamento)
    def update(self, ap, uid, reward):
        self.n[ap][uid] += 1
        q_old = self.q[ap][uid]
        a = self.ema_alpha
        self.q[ap][uid] = (1 - a) * q_old + a * float(reward)

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
def weights_from_q_softmax(ap, qdict, floor=WEIGHT_FLOOR, ceil=WEIGHT_CEIL, temp=TEMP, eps=1e-9):
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
# Loop Federado Hierárquico
# -------------------------
bandit = BanditSelector(ap_users, ema_alpha=0.25)
global_model = build_model()

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    # ACC global no val — base p/ componente "ganho"
    _, acc_val_global = global_model.evaluate(val_ds, verbose=0)

    ap_models_weights = []
    ap_sizes = []

    for ap in range(NUM_AP):
        selected_user_ids = bandit.select_all(ap)

        local_weights_list = []
        local_sizes = []

        for uid in selected_user_ids:
            # cada UE começa do global
            local_model = build_model()
            local_model.set_weights(global_model.get_weights())

            x_u, y_u = user_data[uid]
            local_model.fit(
                x_u, y_u,
                epochs=EPOCHS_LOCAL,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True   # <- ajuda com 100 exemplos/UE
            )

            # Recompensa blend: nível + ganho marginal - penalização de overfit
            _, acc_local = local_model.evaluate(x_u,   y_u,   verbose=0)
            _, acc_val_u = local_model.evaluate(val_ds,        verbose=0)
            reward_level = acc_val_u
            reward_gain  = acc_val_u - acc_val_global
            reward = ALPHA_LEVEL * reward_level + BETA_GAIN * reward_gain \
                     - LAMBDA_OVERFIT * max(0.0, acc_local - acc_val_u)

            bandit.update(ap, uid, reward)

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # 100

        # Converte q -> pesos SUAVES (média≈1)
        mults = weights_from_q_softmax(ap, bandit.q)
        local_sizes_weighted = [n * mults[uid] for n, uid in zip(local_sizes, selected_user_ids)]

        # Agregação no AP
        ap_weights = fedavg(local_weights_list, local_sizes_weighted)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes_weighted))

    # Agregação global (FedAvg puro — sem momentum, mais estável)
    new_global_weights = fedavg(ap_models_weights, ap_sizes)
    global_model.set_weights(new_global_weights)

    # Acompanhar evolução
    loss_r, acc_r = global_model.evaluate(test_ds, verbose=0)
    print(f"   ↳ após agregação dos APs: acc={acc_r*100:.2f}%")

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(test_ds, verbose=2)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")
