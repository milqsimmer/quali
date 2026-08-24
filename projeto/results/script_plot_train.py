import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "figs_treino"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "pure": "RL puro",
    "pirl": "PIRL (PI-reward)",
}

WINDOW = 20  # média móvel em episódios


def read_monitor_file(path):
    """
    Lê monitor.csv do SB3/Monitor.
    O arquivo tem uma primeira linha comentada com '#'.
    """
    return pd.read_csv(path, skiprows=1)


def load_all_monitors(mode):
    pattern = os.path.join(f"runs_{mode}", "seed_*", "monitor.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Nenhum monitor encontrado em: {pattern}")

    dfs = []
    for path in files:
        df = read_monitor_file(path)

        seed_str = os.path.basename(os.path.dirname(path)).replace("seed_", "")
        seed = int(seed_str)

        # colunas padrão do Monitor:
        # r = retorno do episódio
        # l = comprimento do episódio
        # t = tempo de parede
        # + colunas extras de info_keywords
        df["seed"] = seed
        df["episode_idx"] = np.arange(1, len(df) + 1)
        df["timesteps_cumulative"] = df["l"].cumsum()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def aggregate_curves(df, x_col, y_col, max_points=300):
    """
    Agrega curvas de seeds diferentes em uma grade comum de x.
    Retorna x_grid, y_mean, y_std
    """
    seeds = sorted(df["seed"].unique())

    x_min = df.groupby("seed")[x_col].min().max()
    x_max = df.groupby("seed")[x_col].max().min()

    if x_max <= x_min:
        raise ValueError(
            f"Faixa inválida para agregação em {x_col}: {x_min} -> {x_max}"
        )

    x_grid = np.linspace(x_min, x_max, max_points)
    curves = []

    for seed in seeds:
        d = df[df["seed"] == seed].sort_values(x_col)
        x = d[x_col].values
        y = d[y_col].values

        # remove duplicatas de x se necessário
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]

        y_interp = np.interp(x_grid, x_unique, y_unique)
        curves.append(y_interp)

    curves = np.array(curves)
    return x_grid, curves.mean(axis=0), curves.std(axis=0, ddof=1)


# ========= CARREGAMENTO =========
df_pure = load_all_monitors("pure")
df_pirl = load_all_monitors("pirl")

# ========= MÉDIAS MÓVEIS POR SEED =========
for df in [df_pure, df_pirl]:
    df["reward_ma"] = df.groupby("seed")["r"].transform(
        lambda s: moving_average(s, WINDOW)
    )
    df["success_ma"] = df.groupby("seed")["is_success"].transform(
        lambda s: moving_average(s, WINDOW)
    )
    df["final_distance_ma"] = df.groupby("seed")["final_distance"].transform(
        lambda s: moving_average(s, WINDOW)
    )

# ========= FIG 1: REWARD DE TREINO =========
x_pure, y_pure, s_pure = aggregate_curves(df_pure, "timesteps_cumulative", "reward_ma")
x_pirl, y_pirl, s_pirl = aggregate_curves(df_pirl, "timesteps_cumulative", "reward_ma")

fig1, ax1 = plt.subplots()
ax1.plot(x_pure, y_pure, label=LABELS["pure"])
ax1.plot(x_pirl, y_pirl, label=LABELS["pirl"])
ax1.fill_between(x_pure, y_pure - s_pure, y_pure + s_pure, alpha=0.2)
ax1.fill_between(x_pirl, y_pirl - s_pirl, y_pirl + s_pirl, alpha=0.2)
ax1.set_xlabel("Timesteps de treino")
ax1.set_ylabel(f"Retorno por episódio (média móvel, janela={WINDOW})")
ax1.legend()
fig1.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "fig_treino_reward.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig1)

# ========= FIG 2: SUCESSO DE TREINO =========
x_pure, y_pure, s_pure = aggregate_curves(df_pure, "timesteps_cumulative", "success_ma")
x_pirl, y_pirl, s_pirl = aggregate_curves(df_pirl, "timesteps_cumulative", "success_ma")

fig2, ax2 = plt.subplots()
ax2.plot(x_pure, y_pure, label=LABELS["pure"])
ax2.plot(x_pirl, y_pirl, label=LABELS["pirl"])
ax2.fill_between(x_pure, y_pure - s_pure, y_pure + s_pure, alpha=0.2)
ax2.fill_between(x_pirl, y_pirl - s_pirl, y_pirl + s_pirl, alpha=0.2)
ax2.set_xlabel("Timesteps de treino")
ax2.set_ylabel(f"Taxa de sucesso (média móvel, janela={WINDOW})")
ax2.set_ylim(0.0, 1.0)
ax2.legend()
fig2.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "fig_treino_sucesso.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig2)

# ========= FIG 3: DISTÂNCIA FINAL DE TREINO =========
x_pure, y_pure, s_pure = aggregate_curves(
    df_pure, "timesteps_cumulative", "final_distance_ma"
)
x_pirl, y_pirl, s_pirl = aggregate_curves(
    df_pirl, "timesteps_cumulative", "final_distance_ma"
)

fig3, ax3 = plt.subplots()
ax3.plot(x_pure, y_pure, label=LABELS["pure"])
ax3.plot(x_pirl, y_pirl, label=LABELS["pirl"])
ax3.fill_between(x_pure, y_pure - s_pure, y_pure + s_pure, alpha=0.2)
ax3.fill_between(x_pirl, y_pirl - s_pirl, y_pirl + s_pirl, alpha=0.2)
ax3.set_xlabel("Timesteps de treino")
ax3.set_ylabel(f"Distância final (média móvel, janela={WINDOW})")
ax3.legend()
fig3.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "fig_treino_distancia_final.pdf"),
    format="pdf",
    bbox_inches="tight",
)
plt.close(fig3)

print("Gráficos de treino salvos em:", OUT_DIR)
