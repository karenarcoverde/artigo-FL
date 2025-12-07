# import matplotlib.pyplot as plt
# import numpy as np

# # Rodadas
# rounds = np.arange(1, 31)

# # COM RL - todos UEs
# acc_rl_all = np.array([
#     25.51, 33.82, 38.65, 40.24, 42.94, 44.72, 46.24, 47.28, 48.24, 49.24,
#     50.13, 50.87, 51.48, 51.97, 52.41, 52.71, 53.15, 53.45, 53.77, 53.92,
#     54.05, 54.24, 54.48, 54.74, 54.99, 55.24, 55.69, 55.90, 56.05, 56.00
# ])

# # COM RL - top-2 UEs
# acc_rl_top2 = np.array([
#     18.14, 31.86, 38.10, 41.21, 43.10, 44.57, 45.94, 46.49, 47.38, 48.55,
#     49.51, 50.10, 50.81, 51.63, 51.63, 52.10, 51.90, 52.62, 53.23, 53.23,
#     53.29, 53.09, 53.76, 54.26, 54.23, 54.51, 54.99, 55.10, 55.08, 55.42
# ])

# # SEM RL - todos UEs
# acc_norl_all = np.array([
#     25.48, 33.91, 38.53, 40.09, 42.74, 44.77, 45.95, 47.12, 48.09, 48.95,
#     49.77, 50.48, 51.04, 51.52, 52.23, 52.68, 53.05, 52.97, 53.51, 53.78,
#     54.12, 54.29, 54.48, 54.67, 54.89, 55.22, 55.25, 55.48, 55.76, 55.98
# ])

# # SEM RL - top-2 UEs
# acc_norl_top2 = np.array([
#     24.61, 30.72, 36.72, 40.11, 42.85, 44.62, 45.59, 46.67, 47.71, 49.22,
#     48.59, 49.88, 50.78, 50.58, 51.25, 51.89, 52.39, 52.55, 53.39, 53.02,
#     53.07, 53.58, 53.82, 54.34, 54.37, 54.21, 54.60, 54.50, 54.40, 54.59
# ])

# plt.figure(figsize=(10, 4))

# plt.plot(rounds, acc_rl_top2, marker='o', linestyle='-', label='Com reforço (top-2 UEs)')
# plt.plot(rounds, acc_rl_all, marker='s', linestyle='-', label='Com reforço (todos UEs)')
# plt.plot(rounds, acc_norl_all, marker='^', linestyle='-', label='Sem reforço (todos UEs)')
# plt.plot(rounds, acc_norl_top2, marker='D', linestyle='--', label='Sem reforço (top-2 UEs)')

# plt.xlabel('Rodada federada')
# plt.ylabel('Acurácia (%)')
# plt.title('Acurácia ao longo das rodadas federadas\nQuatro esquemas de participação dos UEs')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Esquemas
# labels = [
#     'Com reforço\n(top-2 UEs)',
#     'Com reforço\n(todos UEs)',
#     'Sem reforço\n(todos UEs)',
#     'Sem reforço\n(top-2 UEs)',
# ]

# # FLOPs totais (TFLOPs)
# flops = np.array([2.85, 28.53, 28.53, 2.85])

# x = np.arange(len(labels))

# plt.figure(figsize=(6, 4))
# plt.bar(x, flops)

# plt.xticks(x, labels)
# plt.ylabel('FLOPs totais (TFLOPs)')
# plt.title('Comparação dos FLOPs totais\nQuatro esquemas de participação dos UEs')
# plt.grid(axis='y', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Com reforço\n(top-2 UEs)',
    'Com reforço\n(todos UEs)',
    'Sem reforço\n(todos UEs)',
    'Sem reforço\n(top-2 UEs)',
]

# Índice de Jain de participação
jain = np.array([0.9184, 1.0, 1.0, 0.7679])

x = np.arange(len(labels))

plt.figure(figsize=(6, 4))
plt.bar(x, jain)

plt.xticks(x, labels)
plt.ylabel('Índice de Jain')
plt.ylim(0.7, 1.02)
plt.title('Comparação do índice de justiça (Jain)\nQuatro esquemas de participação dos UEs')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
