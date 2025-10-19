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

# Bandit-weighted FedAvg
VAL_SAMPLES = 5000          # maior val-set -> recompensa mais estável
LAMBDA_OVERFIT = 0.15       # 0.1–0.3 funciona bem
WEIGHT_FLOOR, WEIGHT_CEIL = 0.5, 1.5  # faixa de pesos na agregação por AP

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

# Conjunto de validação compartilhado (maior para reduzir ruído do bandit)
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
# Modelo base (mesmo do baseline)
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
# Bandit (EMA no valor do braço)
# -------------------------
class BanditSelector:
    """
    Mantém q[ap][uid] = EMA da recompensa (recompensa = métrica de utilidade do UE).
    """
    def __init__(self, ap_to_users, ema_alpha=0.25):
        self.ema_alpha = ema_alpha
        self.n = defaultdict(lambda: defaultdict(int))
        self.q = defaultdict(lambda: defaultdict(float))
        self.ap_to_users = ap_to_users

    def select_all(self, ap):
        return self.ap_to_users[ap]  # usamos todos os UEs

    def update(self, ap, uid, reward):
        self.n[ap][uid] += 1
        q_old = self.q[ap][uid]
        a = self.ema_alpha
        self.q[ap][uid] = (1 - a) * q_old + a * float(reward)

# -------------------------
# FedAvg (ponderado)
# -------------------------
def fedavg(weights_list, sizes):
    """ Soma ponderada por n_i / N, camada a camada. """
    N = float(np.sum(sizes))
    new_weights = []
    for layer_ws in zip(*weights_list):
        stack = np.stack(layer_ws, axis=0)  # (num_models, ...)
        coeffs = (np.array(sizes)/N).reshape((-1,) + (1,)*(stack.ndim-1))
        new_weights.append(np.sum(coeffs * stack, axis=0))
    return new_weights

def user_weight_from_q(ap, uid, qdict, floor=WEIGHT_FLOOR, ceil=WEIGHT_CEIL, eps=1e-6):
    qs = [qdict[ap][u] for u in ap_users[ap]]
    q_min, q_max = min(qs), max(qs)
    if q_max - q_min < eps:
        return 1.0
    q_norm = (qdict[ap][uid] - q_min) / (q_max - q_min)
    return floor + (ceil - floor) * q_norm  # peso em [floor, ceil]

# -------------------------
# Loop Federado Hierárquico
# -------------------------
bandit = BanditSelector(ap_users, ema_alpha=0.25)
global_model = build_model()

# Pré-cria datasets para avaliação rápida
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")
    ap_models_weights = []
    ap_sizes = []

    for ap in range(NUM_AP):
        selected_user_ids = bandit.select_all(ap)  # TODOS os UEs

        local_weights_list = []
        local_sizes = []
        user_weights = []

        for uid in selected_user_ids:
            # cada usuário começa do modelo global
            local_model = build_model()
            local_model.set_weights(global_model.get_weights())

            x_u, y_u = user_data[uid]
            local_model.fit(
                x_u, y_u,
                epochs=EPOCHS_LOCAL,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=False
            )

            # Recompensa: preferimos quem generaliza no val
            # (penaliza leve overfit local)
            _, acc_local = local_model.evaluate(x_u, y_u,   verbose=0)
            _, acc_val   = local_model.evaluate(val_ds,     verbose=0)
            reward = acc_val - LAMBDA_OVERFIT * max(0.0, acc_local - acc_val)
            bandit.update(ap, uid, reward)

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # 100 por UE

        # Converte q -> pesos no FedAvg do AP
        for uid in selected_user_ids:
            w_uid = user_weight_from_q(ap, uid, bandit.q)
            user_weights.append(w_uid)

        local_sizes_weighted = [n * w for n, w in zip(local_sizes, user_weights)]
        ap_weights = fedavg(local_weights_list, local_sizes_weighted)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes_weighted))

    # Agregação global
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
