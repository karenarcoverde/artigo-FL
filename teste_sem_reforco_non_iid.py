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
EPOCHS_LOCAL = 5
EPOCHS_GLOBAL = 30
BATCH_SIZE = 64

# Seleção de UEs (sem RL)
USE_ALL_USERS_PER_AP = True   # se False, usa K_PER_AP aleatórios por AP
K_PER_AP = 6

VAL_SAMPLES = 5000            # apenas monitoramento

# Particionamento
USE_DIRICHLET = True          # <<< LIGA/DESLIGA non-IID
ALPHA_DIR = 0.5               # menor => mais não-IID (ex.: 0.3 / 0.1)

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
    """
    Retorna [(x_u, y_u)] para u=0..num_users-1.
    Cada usuário u recebe uma mistura de classes p_u ~ Dir(alpha),
    e uma contagem Multinomial(samples_per_user, p_u).
    """
    rng = np.random.default_rng(seed)

    # índices por classe
    class_indices = {c: np.where(y.flatten() == c)[0] for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # mistura por usuário ~ Dirichlet
    P = rng.dirichlet(alpha * np.ones(num_classes), size=num_users)

    # contagens Multinomial(S, p_u)
    counts = np.zeros((num_users, num_classes), dtype=int)
    for u in range(num_users):
        counts[u] = rng.multinomial(samples_per_user, P[u])

    # garantir que não exceda disponibilidade por classe (ajuste proporcional)
    for c in range(num_classes):
        need = int(counts[:, c].sum())
        have = int(len(class_indices[c]))
        if need <= have:
            continue
        # escala para caber
        scale = have / max(1, need)
        scaled = np.floor(counts[:, c] * scale).astype(int)
        rest = have - int(scaled.sum())
        # distribui as sobras segundo os maiores resíduos
        residuals = (counts[:, c] * scale) - scaled
        order = np.argsort(-residuals)
        for i in range(rest):
            scaled[order[i]] += 1
        counts[:, c] = scaled

    # monta os índices por usuário
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

    # datasets por usuário (pode ficar com < samples_per_user se faltou alguma classe)
    user_data = []
    for u in range(num_users):
        idxs = np.array(user_indices[u], dtype=int)
        user_data.append((x[idxs], y[idxs]))
    return user_data

# -------------------------
# Dados
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
num_classes = 10

if USE_DIRICHLET:
    # Non-IID forte/leve conforme ALPHA_DIR
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

# Val set (apenas monitoramento)
VAL_SAMPLES = min(VAL_SAMPLES, len(x_test))
x_val = x_test[:VAL_SAMPLES]; y_val = y_test[:VAL_SAMPLES]

# Índices de usuários por AP
ap_users = { ap: list(range(ap*USERS_PER_AP, (ap+1)*USERS_PER_AP)) for ap in range(NUM_AP) }

# -------------------------
# Seleção de usuários (sem RL)
# -------------------------
rng = np.random.default_rng(SEED)
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
            local_sizes.append(len(x_u))  # pode variar levemente no Dirichlet

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
