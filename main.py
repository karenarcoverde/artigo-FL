import numpy as np
from teste import reforco

def main():
    SEEDS = list(range(10))

    jain_vals = []
    acc_vals = []

    for SEED in SEEDS:
        jain_part, acc = reforco(SEED)

        # se forem escalares:
        jain_vals.append(jain_part)
        acc_vals.append(acc)

        # se "acc" for uma lista/array por época e você quiser só a final, use:
        # acc_vals.append(acc[-1])

    jain_vals = np.array(jain_vals, dtype=float)
    acc_vals = np.array(acc_vals, dtype=float)

    # média e desvio padrão (std populacional: ddof=0; amostral: ddof=1)
    jain_mean = np.mean(jain_vals)
    jain_std = np.std(jain_vals, ddof=1)

    acc_mean = np.mean(acc_vals)
    acc_std = np.std(acc_vals, ddof=1)

    print("Jain_part -> média =", jain_mean, " | desvio padrão =", jain_std)
    print("Acc       -> média =", acc_mean,  " | desvio padrão =", acc_std)

if __name__ == "__main__":
    main()
