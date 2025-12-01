import matplotlib.pyplot as plt
import numpy as np

# 30 rodadas
rounds = np.arange(1, 31)

# 1) COM reforço, TODOS os UEs
acc_ref_all = [
    25.00, 34.33, 39.14, 41.69, 43.15, 44.87, 46.26, 47.58, 48.35, 49.45,
    49.96, 50.61, 51.29, 51.67, 52.23, 52.56, 53.06, 53.34, 53.57, 53.51,
    53.80, 53.73, 53.90, 54.00, 54.09, 54.32, 54.57, 54.58, 54.69, 54.83
]

# 2) COM reforço, top-4 UEs (por AP)
acc_ref_top4 = [
    21.42, 32.97, 37.98, 40.00, 42.28, 44.59, 44.90, 46.10, 47.35, 48.28,
    49.14, 50.07, 50.15, 50.71, 51.29, 52.02, 51.88, 52.52, 52.62, 52.83,
    53.44, 53.34, 53.65, 53.92, 53.89, 53.99, 53.83, 54.33, 54.26, 54.39
]

# 3) SEM reforço, TODOS os UEs
acc_no_ref_all = [
    24.72, 34.89, 39.50, 41.10, 43.34, 44.68, 46.24, 47.54, 48.49, 49.25,
    50.15, 50.72, 51.04, 51.75, 52.21, 52.43, 52.75, 53.22, 53.38, 53.70,
    53.93, 53.87, 53.88, 54.06, 54.22, 54.23, 54.45, 54.48, 54.51, 54.68
]

plt.figure(figsize=(10,4))

# curvas
plt.plot(rounds, acc_ref_top4, marker='o', linestyle='-', label='Com reforço (top-4 UEs)')
plt.plot(rounds, acc_ref_all,  marker='s', linestyle='-', label='Com reforço (todos UEs)')
plt.plot(rounds, acc_no_ref_all, marker='^', linestyle='-', label='Sem reforço (todos UEs)')

plt.xlabel('Rodada federada')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia ao longo das rodadas federadas\nTrês esquemas de participação dos UEs')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # FLOPs totais em TFLOPs (pelos seus relatórios)
# flops_ref_top4   = 4.56   # com reforço, top-4 UEs
# flops_ref_all    = 11.41  # com reforço, todos UEs
# flops_no_ref_all = 11.41  # sem reforço, todos UEs

# labels = [
#     'Com reforço\n(top-4 UEs)',
#     'Com reforço\n(todos UEs)',
#     'Sem reforço\n(todos UEs)'
# ]
# values = [flops_ref_top4, flops_ref_all, flops_no_ref_all]

# x = np.arange(len(labels))

# plt.figure(figsize=(6,4))
# plt.bar(x, values)
# plt.xticks(x, labels)
# plt.ylabel('FLOPs totais (TFLOPs)')
# plt.title('Comparação dos FLOPs totais\nTrês esquemas de participação dos UEs')
# plt.grid(axis='y', linestyle='--', alpha=0.4)

# plt.tight_layout()
# plt.show()
