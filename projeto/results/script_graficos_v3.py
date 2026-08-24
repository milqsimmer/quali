import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES GERAIS =========
# Por padrão, lê todos os CSVs de avaliação oficial gerados por run_experiments_seeds.py,
# que têm o padrão: results/eval_official_seed<seed>_<seed>.csv
CSV_GLOB = r"results/eval_official_seed*_*.csv"
OUT_DIR = r"figs_resultados"
TABELAS_DIR = r"tabelas"

# ========= LABELS =========
LABELS = {
    "pure": "RL puro (baseline)",
    "pirl": "PIRL (PI-reward)",
}

# ========= PREPARO DE PASTAS =========
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABELAS_DIR, exist_ok=True)


# ========= LEITURA DO CSV (tenta ',' e ';') =========
def read_csv_auto(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def extract_seed_from_filename(path):
    """Extrai a seed do nome do arquivo de avaliação.

    Exemplos aceitos:
    - eval_official_seed0_0.csv -> 0
    - eval_official_seed3_3.csv -> 3

    Mantém compatibilidade com o formato antigo "eval_results_0.csv".
    """
    name = os.path.basename(path)

    # Novo padrão: eval_official_seed<seed>_<seed>.csv
    m_new = re.search(r"eval_official_seed(\d+)_\d+\.csv$", name)
    if m_new:
        return int(m_new.group(1))

    # Padrão legado: eval_results_<seed>.csv
    m_old = re.search(r"eval_results_(\d+)\.csv$", name)
    if m_old:
        return int(m_old.group(1))

    raise ValueError(f"Não foi possível extrair a seed do nome do arquivo: {name}")


def normalize_mode(value):
    """
    Normaliza os nomes dos modos para:
    - pure
    - pirl
    """
    if pd.isna(value):
        return np.nan

    v = str(value).strip().lower()

    pure_aliases = {
        "pure",
        "rl",
        "baseline",
        "rl puro",
        "rl_puro",
        "pure_rl",
    }
    pirl_aliases = {
        "pirl",
        "pi-reward",
        "pi_reward",
        "physics-informed",
        "physics_informed",
        "physics informed",
    }

    if v in pure_aliases:
        return "pure"
    if v in pirl_aliases:
        return "pirl"

    # fallback: procura por substring
    if "pure" in v or "baseline" in v:
        return "pure"
    if "pirl" in v or "pi-reward" in v or "physics" in v:
        return "pirl"

    return v


def mean_std(series):
    return series.mean(), series.std(ddof=1)


def resumo_metrica(df_mode):
    succ = df_mode["success"].mean()  # taxa (0..1)
    d_mean, d_std = mean_std(df_mode["final_distance"])
    l_mean, l_std = mean_std(df_mode["length"])
    r_mean, r_std = mean_std(df_mode["return"])

    tau_mean = tau_std = np.nan
    energy_mean = energy_std = np.nan
    if "mean_tau_sum" in df_mode.columns:
        tau_mean, tau_std = mean_std(df_mode["mean_tau_sum"])
    if "energy" in df_mode.columns:
        energy_mean, energy_std = mean_std(df_mode["energy"])

    return (
        succ,
        d_mean,
        d_std,
        l_mean,
        l_std,
        r_mean,
        r_std,
        tau_mean,
        tau_std,
        energy_mean,
        energy_std,
    )


# ========= LEITURA DE TODOS OS CSVs =========
csv_files = sorted(glob.glob(CSV_GLOB))
if not csv_files:
    raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {CSV_GLOB}")

dfs = []
for path in csv_files:
    temp = read_csv_auto(path)
    temp["seed"] = extract_seed_from_filename(path)
    temp["source_file"] = os.path.basename(path)
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# ========= CHECAGEM MÍNIMA =========
required_cols = ["mode", "episode", "success", "final_distance", "return", "length"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Colunas faltando no CSV: {missing}. Precisam ser {required_cols}"
    )

# ========= NORMALIZAÇÃO =========
df["mode_norm"] = df["mode"].apply(normalize_mode)

# Mantém apenas os dois modos de interesse
df = df[df["mode_norm"].isin(["pure", "pirl"])].copy()

# ========= TIPOS NUMÉRICOS =========
num_cols = ["episode", "success", "final_distance", "return", "length", "seed"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Remove linhas com NaN em colunas essenciais
df = df.dropna(
    subset=["mode_norm", "seed", "success", "final_distance", "return", "length"]
).copy()

# ========= RESUMO GERAL =========
summary_rows = []
for mode in ["pure", "pirl"]:
    df_mode = df[df["mode_norm"] == mode]
    (
        succ,
        d_mean,
        d_std,
        l_mean,
        l_std,
        r_mean,
        r_std,
        tau_mean,
        tau_std,
        energy_mean,
        energy_std,
    ) = resumo_metrica(df_mode)

    summary_rows.append(
        {
            "mode": LABELS[mode],
            "success_rate_mean": succ,
            "final_distance_mean": d_mean,
            "final_distance_std": d_std,
            "length_mean": l_mean,
            "length_std": l_std,
            "return_mean": r_mean,
            "return_std": r_std,
            "tau_effort_mean": tau_mean,
            "tau_effort_std": tau_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "n_episodes": len(df_mode),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(TABELAS_DIR, "resumo_geral_todos_csvs.csv"),
    index=False,
    encoding="utf-8-sig",
)

# ========= RESUMO POR SEED =========
agg_dict = {
    "success_rate": ("success", "mean"),
    "final_distance": ("final_distance", "mean"),
    "episode_length": ("length", "mean"),
    "episode_return": ("return", "mean"),
    "n_episodes": ("episode", "count"),
}

if "mean_tau_sum" in df.columns:
    agg_dict["tau_effort"] = ("mean_tau_sum", "mean")
if "energy" in df.columns:
    agg_dict["energy"] = ("energy", "mean")

seed_summary = (
    df.groupby(["seed", "mode_norm"], as_index=False)
    .agg(**agg_dict)
    .sort_values(["seed", "mode_norm"])
)

seed_summary["mode_label"] = seed_summary["mode_norm"].map(LABELS)

seed_summary.to_csv(
    os.path.join(TABELAS_DIR, "resumo_por_seed.csv"),
    index=False,
    encoding="utf-8-sig",
)

# ========= DADOS PARA PLOTS =========
data_pure_dist = df[df["mode_norm"] == "pure"]["final_distance"].values
data_pirl_dist = df[df["mode_norm"] == "pirl"]["final_distance"].values

data_pure_len = df[df["mode_norm"] == "pure"]["length"].values
data_pirl_len = df[df["mode_norm"] == "pirl"]["length"].values

data_pure_succ = df[df["mode_norm"] == "pure"]["success"].values
data_pirl_succ = df[df["mode_norm"] == "pirl"]["success"].values

# ========= FIGURA 1: BOXPLOT (SUCCESS + DISTÂNCIA) =========

# calcular métricas por seed
seed_metrics = (
    df.groupby(["seed", "mode_norm"])
    .agg(
        success_rate=("success", "mean"),
        final_distance=("final_distance", "mean"),
    )
    .reset_index()
)

pure_seed = seed_metrics[seed_metrics["mode_norm"] == "pure"]
pirl_seed = seed_metrics[seed_metrics["mode_norm"] == "pirl"]

fig1, (ax1, ax2) = plt.subplots(1, 2)

# --- (a) Success ---
ax1.boxplot(
    [pure_seed["success_rate"], pirl_seed["success_rate"]],
    tick_labels=[LABELS["pure"], LABELS["pirl"]],
)
ax1.set_ylabel("Taxa de sucesso")

# --- (b) Distância ---
ax2.boxplot(
    [pure_seed["final_distance"], pirl_seed["final_distance"]],
    tick_labels=[LABELS["pure"], LABELS["pirl"]],
)
ax2.set_ylabel("Distância final (m)")

fig1.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_boxplot_combined.pdf"))
plt.close()


# ========= FIGURA 2: PERFORMANCE POR SEED =========

fig2, (ax1, ax2) = plt.subplots(1, 2)

# pivot
pivot_succ = seed_metrics.pivot(
    index="seed", columns="mode_norm", values="success_rate"
)
pivot_dist = seed_metrics.pivot(
    index="seed", columns="mode_norm", values="final_distance"
)

seeds = pivot_succ.index

# --- (a) Success ---
ax1.plot(seeds, pivot_succ["pure"], marker="o", label=LABELS["pure"])
ax1.plot(seeds, pivot_succ["pirl"], marker="o", label=LABELS["pirl"])
ax1.set_xlabel("Seed")
ax1.set_ylabel("Taxa de sucesso")
ax1.set_ylim(0, 1)
ax1.legend()

# --- (b) Distância ---
ax2.plot(seeds, pivot_dist["pure"], marker="o", label=LABELS["pure"])
ax2.plot(seeds, pivot_dist["pirl"], marker="o", label=LABELS["pirl"])
ax2.set_xlabel("Seed")
ax2.set_ylabel("Distância final (m)")
ax2.legend()

fig2.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_performance_seed.pdf"))
plt.close()

print("Arquivos CSV lidos:")
for f in csv_files:
    print(" -", f)

print("\nFiguras salvas em:", OUT_DIR)
print("Tabelas salvas em:", TABELAS_DIR)

# ========= FIGURA 3: RESUMO (SUCESSO, DISTÂNCIA, TORQUE, ENERGIA) =========

has_tau = "tau_effort" in seed_summary.columns
has_energy = "energy" in seed_summary.columns

if has_tau or has_energy:
    # Preparar dados por seed para cada métrica
    pure_seed = seed_summary[seed_summary["mode_norm"] == "pure"]
    pirl_seed = seed_summary[seed_summary["mode_norm"] == "pirl"]

    metrics = [
        ("success_rate", "Taxa de sucesso", (0.0, 1.0)),
        ("final_distance", "Distância final (m)", None),
    ]
    if has_tau:
        metrics.append(("tau_effort", "Esforço médio em torque", None))
    if has_energy:
        metrics.append(("energy", "Energia média por episódio", None))

    n_plots = len(metrics)
    n_rows = 2
    n_cols = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(2)
    tick_labels = [LABELS["pure"], LABELS["pirl"]]

    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        pure_vals = pure_seed[col].values
        pirl_vals = pirl_seed[col].values

        means = [np.nanmean(pure_vals), np.nanmean(pirl_vals)]
        stds = [np.nanstd(pure_vals, ddof=1), np.nanstd(pirl_vals, ddof=1)]

        ax.bar(x, means, yerr=stds, tick_label=tick_labels, capsize=5)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig3_resumo_barras.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ========= FIGURA 4: TRADE-OFF ENERGIA vs SUCESSO POR SEED =========
    if has_energy:
        fig4, ax4 = plt.subplots()

        for mode in ["pure", "pirl"]:
            sub = seed_summary[seed_summary["mode_norm"] == mode]
            xs = sub["energy"].values
            ys = sub["success_rate"].values
            seeds_local = sub["seed"].values

            ax4.scatter(xs, ys, label=LABELS[mode])

            # adiciona o número da seed próximo a cada ponto
            # pequeno deslocamento vertical para evitar sobreposição com o marcador
            for x, y, s in zip(xs, ys, seeds_local):
                ax4.text(x, y + 0.02, str(s), fontsize=8, ha="center", va="bottom")

        ax4.set_xlabel("Energia média por episódio")
        ax4.set_ylabel("Taxa de sucesso")
        ax4.set_ylim(0.0, 1.0)
        ax4.legend()
        # pequena nota explicativa para os rótulos numéricos
        ax4.text(
            0.02,
            0.02,
            "Números indicam seeds",
            transform=ax4.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        fig4.tight_layout()
        fig4_path = os.path.join(OUT_DIR, "fig4_tradeoff_energia_sucesso.pdf")
        plt.savefig(fig4_path, format="pdf", bbox_inches="tight")
        plt.close(fig4)

    # ========= FIGURA 5: RESUMO (SUCESSO + DISTÂNCIA) =========
    # Mesma ideia da figura 3, mas apenas com métricas de tarefa.
    metrics_sd = [
        ("success_rate", "Taxa de sucesso", (0.0, 1.0)),
        ("final_distance", "Distância final (m)", None),
    ]

    n_plots_sd = len(metrics_sd)
    fig_sd, axes_sd = plt.subplots(1, n_plots_sd, figsize=(5 * n_plots_sd, 4))
    axes_sd = np.atleast_1d(axes_sd).flatten()

    x = np.arange(2)
    tick_labels = [LABELS["pure"], LABELS["pirl"]]

    for ax, (col, ylabel, ylim) in zip(axes_sd, metrics_sd):
        pure_vals = pure_seed[col].values
        pirl_vals = pirl_seed[col].values

        means = [np.nanmean(pure_vals), np.nanmean(pirl_vals)]
        stds = [np.nanstd(pure_vals, ddof=1), np.nanstd(pirl_vals, ddof=1)]

        ax.bar(x, means, yerr=stds, tick_label=tick_labels, capsize=5)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig_sd.tight_layout()
    fig_sd_path = os.path.join(OUT_DIR, "fig5_resumo_sucesso_distancia.pdf")
    plt.savefig(fig_sd_path, format="pdf", bbox_inches="tight")
    plt.close(fig_sd)

    # ========= FIGURA 6: RESUMO (TORQUE + ENERGIA) =========
    # Apenas se as duas métricas estiverem disponíveis.
    if has_tau and has_energy:
        metrics_te = [
            ("tau_effort", "Esforço médio em torque", None),
            ("energy", "Energia média por episódio", None),
        ]

        fig_te, axes_te = plt.subplots(1, 2, figsize=(10, 4))
        axes_te = np.atleast_1d(axes_te).flatten()

        for ax, (col, ylabel, ylim) in zip(axes_te, metrics_te):
            pure_vals = pure_seed[col].values
            pirl_vals = pirl_seed[col].values

            means = [np.nanmean(pure_vals), np.nanmean(pirl_vals)]
            stds = [np.nanstd(pure_vals, ddof=1), np.nanstd(pirl_vals, ddof=1)]

            ax.bar(x, means, yerr=stds, tick_label=tick_labels, capsize=5)
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)

        fig_te.tight_layout()
        fig_te_path = os.path.join(OUT_DIR, "fig6_resumo_torque_energia.pdf")
        plt.savefig(fig_te_path, format="pdf", bbox_inches="tight")
        plt.close(fig_te)
