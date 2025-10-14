import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# -------------------------
# Configurações iniciais
# -------------------------
NUM_AP = 4                # Número de Access Points
USERS_PER_AP = 10         # Usuários por AP
TOTAL_USERS = NUM_AP * USERS_PER_AP
EPOCHS_LOCAL = 2
EPOCHS_GLOBAL = 3
BATCH_SIZE = 64

# -------------------------
# Carregar CIFAR-10 e dividir entre usuários
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reduzir base para facilitar simulação
x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=TOTAL_USERS * 100, stratify=y_train)

# Dividir para os usuários
user_data = []
for i in range(TOTAL_USERS):
    x_user = x_train[i*100:(i+1)*100]
    y_user = y_train[i*100:(i+1)*100]
    user_data.append((x_user, y_user))

# -------------------------
# Modelo base (CNN)
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
# Simular métrica de canal e seleção de usuários
# -------------------------
def select_users_by_ap():
    channel_quality = np.random.rand(NUM_AP, USERS_PER_AP)  # Simula qualidade de canal
    selected_users = np.argmax(channel_quality, axis=1)      # Um usuário com melhor canal por AP
    user_ids = [ap * USERS_PER_AP + selected_users[ap] for ap in range(NUM_AP)]
    return user_ids

# -------------------------
# Federated Learning Loop
# -------------------------
global_model = build_model()

for round in range(EPOCHS_GLOBAL):
    print(f"\n🔁 Rodada Federada {round+1}")
    
    selected_user_ids = select_users_by_ap()
    weights = []
    
    for uid in selected_user_ids:
        local_model = build_model()
        local_model.set_weights(global_model.get_weights())
        
        x_u, y_u = user_data[uid]
        local_model.fit(x_u, y_u, epochs=EPOCHS_LOCAL, batch_size=BATCH_SIZE, verbose=0)
        weights.append(local_model.get_weights())
    
    # Média dos pesos (FedAvg)
    new_weights = []
    for layer_weights in zip(*weights):
        new_weights.append(np.mean(layer_weights, axis=0))
    
    global_model.set_weights(new_weights)

# -------------------------
# Avaliação final
# -------------------------
loss, acc = global_model.evaluate(x_test, y_test, verbose=2)
print(f"\n🎯 Acurácia final do modelo global: {acc * 100:.2f}%")