import os, random
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

# Se False, subamostra k usuários por AP por rodada
USE_ALL_USERS_PER_AP = True
K_PER_AP = 2  # usado se USE_ALL_USERS_PER_AP = False

# RNG único para todo o script (evita repetir seleção)
rng = np.random.default_rng(SEED)

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
# Agregações (FedAvg)
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

# -------------------------
# Seleção de usuários por AP (por rodada)
# -------------------------
def select_users_for_ap(ap, round_idx):
    users = ap_users[ap]
    if USE_ALL_USERS_PER_AP:
        return users
    # Subamostragem (aleatória e justa)
    if K_PER_AP >= len(users):
        return users
    # escolhe K_PER_AP usuários diferentes a cada rodada (random sem reposição)
    return list(rng.choice(users, size=K_PER_AP, replace=False))

# -------------------------
# Loop Federado Hierárquico: Usuários -> AP -> Servidor
# -------------------------
global_model = build_model()

for r in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {r+1}")
    ap_models_weights = []
    ap_sizes = []

    # 1) Cada AP coordena seus usuários e faz uma agregação local
    for ap in range(NUM_AP):
        selected_user_ids = select_users_for_ap(ap, r)
        local_weights_list = []
        local_sizes = []

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
                shuffle=False,
                workers=1, use_multiprocessing=False
            )

            local_weights_list.append(local_model.get_weights())
            local_sizes.append(len(x_u))  # aqui: 100 por usuário

        # Agregação no AP (modelo do AP)
        ap_weights = fedavg(local_weights_list, local_sizes)
        ap_models_weights.append(ap_weights)
        ap_sizes.append(np.sum(local_sizes))  # total de amostras desse AP

    # 2) Servidor agrega os modelos dos APs (FedAvg global)
    new_global_weights = fedavg(ap_models_weights, ap_sizes)
    global_model.set_weights(new_global_weights)

    # (Opcional) Acompanhar evolução
    loss_r, acc_r = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"   ↳ após agregação dos APs: acc={acc_r*100:.2f}%")

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(x_test, y_test, verbose=2)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")
