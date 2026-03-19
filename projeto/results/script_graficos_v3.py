import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES GERAIS =========
CSV_GLOB = r"eval_results_*.csv"
OUT_DIR = r"figs_resultados"
TABELAS_DIR = r"tabelas"

# ========= LABELS =========
LABELS = {
    "pure": "RL puro",
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
    """
    Extrai a seed do nome do arquivo:
    eval_results_0.csv -> 0
    """
    name = os.path.basename(path)
    match = re.search(r"eval_results_(\d+)\.csv$", name)
    if match:
        return int(match.group(1))
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
    return succ, d_mean, d_std, l_mean, l_std, r_mean, r_std


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
    succ, d_mean, d_std, l_mean, l_std, r_mean, r_std = resumo_metrica(df_mode)

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
seed_summary = (
    df.groupby(["seed", "mode_norm"], as_index=False)
    .agg(
        success_rate=("success", "mean"),
        final_distance=("final_distance", "mean"),
        episode_length=("length", "mean"),
        episode_return=("return", "mean"),
        n_episodes=("episode", "count"),
    )
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
