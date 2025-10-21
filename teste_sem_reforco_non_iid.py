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

# Grau de não-IID (Dirichlet)
ALPHA_DIRICHLET = 0.3         # ↓ => mais não-IID, ↑ => mais IID (ex.: 1.0)

# -------------------------
# Dados (NÃO-IID com Dirichlet)
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

# (opcional) limitar o pool para bater com TOTAL_USERS * SAMPLES_PER_USER
TOTAL_TRAIN = TOTAL_USERS * SAMPLES_PER_USER
x_pool, _, y_pool, _ = train_test_split(
    x_train, y_train,
    train_size=TOTAL_TRAIN,
    stratify=y_train,
    random_state=SEED,
    shuffle=True
)

def build_dirichlet_non_iid_splits(x, y, total_users, samples_per_user, alpha=0.3, seed=SEED):
    """
    Distribui (x,y) em 'total_users' usuários com viés de classe por Dirichlet.
    Respeita capacidade = samples_per_user por usuário.
    """
    rng = np.random.default_rng(seed)
    y_flat = y.reshape(-1)                    # (N,)
    num_classes = int(np.max(y_flat) + 1)
    N = len(y_flat)
    cap = np.array([samples_per_user]*total_users, dtype=int)

    # índices por classe
    class_to_idx = {c: np.where(y_flat == c)[0] for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(class_to_idx[c])

    # listas por usuário
    user_indices = [list() for _ in range(total_users)]
    leftover = []

    for c in range(num_classes):
        idxs = class_to_idx[c]
        n_c = len(idxs)
        if n_c == 0:
            continue

        # amostra proporções por usuário
        p = rng.dirichlet(alpha * np.ones(total_users, dtype=np.float32))
        alloc = np.floor(p * n_c).astype(int)

        # distribuir sobras por maiores frações
        remainder = n_c - np.sum(alloc)
        if remainder > 0:
            frac_order = np.argsort(-(p - alloc / max(1, n_c)))
            for u in frac_order[:remainder]:
                alloc[u] += 1

        # alocar respeitando capacidade
        start = 0
        order_users = list(range(total_users))
        rng.shuffle(order_users)
        for u in order_users:
            k = int(alloc[u])
            if k <= 0:
                continue
            take = min(k, cap[u])
            if take > 0:
                user_indices[u].extend(idxs[start:start+take])
                start += take
                cap[u] -= take
            rem = k - take
            if rem > 0:
                leftover.extend(idxs[start:start+rem])
                start += rem

        if start < n_c:
            leftover.extend(idxs[start:])

    # preencher quem ficou faltando com leftovers
    rng.shuffle(leftover)
    li = 0
    for u in range(total_users):
        need = cap[u]
        if need <= 0:
            continue
        got = min(need, len(leftover) - li)
        if got > 0:
            user_indices[u].extend(leftover[li:li+got])
            li += got
            cap[u] -= got

    # truncar/preencher para exatamente samples_per_user por usuário
    for u in range(total_users):
        arr = np.array(user_indices[u], dtype=np.int64)
        rng.shuffle(arr)
        if len(arr) < samples_per_user:
            need = samples_per_user - len(arr)
            # fallback: amostrar com reposição do próprio x
            extra = rng.choice(len(x), size=need, replace=True)
            arr = np.concatenate([arr, extra])
        elif len(arr) > samples_per_user:
            arr = arr[:samples_per_user]
        user_indices[u] = arr

    # montar tuplas (x_u, y_u)
    user_data = []
    for u in range(total_users):
        idxs = user_indices[u]
        x_u = x[idxs]
        y_u = y[idxs]                      # mantém shape (n,1) para sparse_categorical
        user_data.append((x_u, y_u))
    return user_data

# Constrói splits não-IID no pool
user_data = build_dirichlet_non_iid_splits(
    x_pool, y_pool,
    total_users=TOTAL_USERS,
    samples_per_user=SAMPLES_PER_USER,
    alpha=ALPHA_DIRICHLET,
    seed=SEED
)

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
    return list(np.random.default_rng(SEED + round_idx + ap).choice(users, size=K_PER_AP, replace=False))

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
