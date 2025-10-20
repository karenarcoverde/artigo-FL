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

SAMPLES_PER_USER = 300        # antes era 100
EPOCHS_LOCAL = 5              # com 300 amostras/UE, pode testar 7–10
EPOCHS_GLOBAL = 30            # pode testar 40
BATCH_SIZE = 64

# Seleção de UEs (sem RL)
USE_ALL_USERS_PER_AP = True   # se False, usa K_PER_AP aleatórios por AP
K_PER_AP = 6

VAL_SAMPLES = 5000            # só para acompanhar global (não influencia seleção)

# -------------------------
# Dados (IID com mais amostras por UE)
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

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

x_val = x_test[:VAL_SAMPLES]
y_val = y_test[:VAL_SAMPLES]

# Índices de usuários por AP
ap_users = { ap: list(range(ap*USERS_PER_AP, (ap+1)*USERS_PER_AP)) for ap in range(NUM_AP) }

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
# Seleção de usuários (sem RL)
# -------------------------
def select_users_for_ap(ap, round_idx):
    users = ap_users[ap]
    if USE_ALL_USERS_PER_AP or K_PER_AP >= len(users):
        return users
    return list(rng.choice(users, size=K_PER_AP, replace=False))

# -------------------------
# Loop Federado Hierárquico
# -------------------------
global_model = build_model()

val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")

    ap_models_weights = []
    ap_sizes = []

    for ap in range(NUM_AP):
        selected_user_ids = select_users_for_ap(ap, r)

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
                epochs=EPOCHS_LOCAL,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True
            )

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # SAMPLES_PER_USER (ex.: 300)

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
