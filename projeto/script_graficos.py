# script_graficos_v46.py
import os
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES =========
# Pode ser .zip (recomendado) ou um .csv já concatenado.
INPUT_PATH = r"results/v4.6/eval__v4.6-α=2e-5__k=10.zip"

OUT_DIR = r"figs_resultados"
TABELAS_DIR = r"tabelas"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABELAS_DIR, exist_ok=True)

# Comparação principal (baseline vs PI-power a2e-5)
# Se no futuro você tiver outros PI-metrics, dá para estender.
LABELS = {
    "pure": "Baseline (RL puro)",
    "pi_power": r"PI-power ($\alpha=2\times10^{-5}$)",
    "pi_tau": r"PI-$\|\tau\|_1$",
}


# ========= LEITURA =========
def read_csv_auto_from_bytes(b: bytes) -> pd.DataFrame:
    """Lê CSV tentando separador ',' e ';'."""
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        return pd.read_csv(io.BytesIO(b), sep=";")


def read_input(path: str) -> pd.DataFrame:
    if path.lower().endswith(".zip"):
        dfs = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                # pega somente os CSVs de episódios (não pega summaries, nem ALL__)
                if (
                    name.endswith(".csv")
                    and "__evalbase" in name
                    and "ALL__" not in name
                ):
                    df = read_csv_auto_from_bytes(z.read(name))
                    df["_source_file"] = name
                    dfs.append(df)
        if not dfs:
            raise ValueError(
                "Não encontrei CSVs de avaliação dentro do ZIP. Verifique o arquivo."
            )
        return pd.concat(dfs, ignore_index=True)

    # fallback: csv único
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


df = read_input(INPUT_PATH)

# ========= NORMALIZAÇÃO DE COLUNAS =========
rename_map = {
    "final_dist": "final_distance",
    "ep_len": "length",
    "ret": "return",
}
df.rename(
    columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
)

required_cols = [
    "exp_id",
    "train_seed",
    "episode",
    "success",
    "final_distance",
    "return",
    "length",
    "use_pi_reward",
    "pi_metric",
    "alpha_pi",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colunas faltando: {missing}\nDisponíveis: {df.columns.tolist()}")


# ========= CONVERSÃO NUMÉRICA ROBUSTA (inclusive vírgula decimal) =========
def to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


num_cols = [
    "train_seed",
    "episode",
    "success",
    "final_distance",
    "return",
    "length",
    "use_pi_reward",
    "alpha_pi",
]
for c in num_cols:
    df[c] = to_numeric_safe(df[c])

# Algumas colunas extras (se existirem) — úteis p/ torque/power
for c in ["mean_tau_l1", "mean_power_abs"]:
    if c in df.columns:
        df[c] = to_numeric_safe(df[c])

df["pi_metric"] = df["pi_metric"].astype(str).str.lower().str.strip()
df["exp_id"] = df["exp_id"].astype(str)

df = df.dropna(
    subset=[
        "train_seed",
        "episode",
        "success",
        "final_distance",
        "return",
        "length",
        "use_pi_reward",
    ]
)


# ========= DEFINIÇÃO DE CONDIÇÃO (baseline vs PI) =========
def classify_condition(row) -> str:
    if row["use_pi_reward"] < 0.5:
        return "pure"
    # PI-reward
    if "power" in row["pi_metric"]:
        return "pi_power"
    if "tau" in row["pi_metric"]:
        return "pi_tau"
    # fallback
    return "pi_power"


df["cond"] = df.apply(classify_condition, axis=1)
df["cond_label"] = df["cond"].map(lambda k: LABELS.get(k, k))

# Mantém somente as condições relevantes para o deck (pure vs pi_power)
keep = df["cond"].isin(["pure", "pi_power"])
df = df[keep].copy()

# ========= RESUMOS =========
# Resumo por seed (essa é a unidade experimental mais defensável)
agg_cols = {
    "success": "mean",
    "final_distance": "mean",
    "length": "mean",
    "return": "mean",
}
if "mean_tau_l1" in df.columns:
    agg_cols["mean_tau_l1"] = "mean"
if "mean_power_abs" in df.columns:
    agg_cols["mean_power_abs"] = "mean"

by_seed = (
    df.groupby(["cond", "train_seed"], as_index=False)
    .agg(agg_cols)
    .rename(
        columns={
            "success": "success_rate",
            "final_distance": "mean_final_dist",
            "length": "mean_ep_len",
            "return": "mean_return",
        }
    )
)

by_seed["cond_label"] = by_seed["cond"].map(lambda k: LABELS.get(k, k))
by_seed.to_csv(os.path.join(TABELAS_DIR, "resumo_por_seed.csv"), index=False)


# Resumo global (média e std ACROSS seeds — mais correto do que across episódios)
def mean_std(x: pd.Series):
    return float(x.mean()), float(x.std(ddof=1))


overall_rows = []
for cond, sub in by_seed.groupby("cond"):
    row = {"cond": cond, "cond_label": LABELS.get(cond, cond)}
    row["success_mean"], row["success_std"] = mean_std(sub["success_rate"])
    row["finaldist_mean"], row["finaldist_std"] = mean_std(sub["mean_final_dist"])
    row["len_mean"], row["len_std"] = mean_std(sub["mean_ep_len"])
    row["return_mean"], row["return_std"] = mean_std(sub["mean_return"])
    if "mean_tau_l1" in sub.columns:
        row["tau_mean"], row["tau_std"] = mean_std(sub["mean_tau_l1"])
    if "mean_power_abs" in sub.columns:
        row["power_mean"], row["power_std"] = mean_std(sub["mean_power_abs"])
    overall_rows.append(row)

overall = pd.DataFrame(overall_rows)
overall.to_csv(
    os.path.join(TABELAS_DIR, "resumo_overall_media_std_seeds.csv"), index=False
)


# ========= HELPERS DE PLOT =========
def save_fig(fig, name_base: str):
    png = os.path.join(OUT_DIR, f"{name_base}.png")
    pdf = os.path.join(OUT_DIR, f"{name_base}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ========= FIG 1: Boxplot distância final (episódios) =========
fig, ax = plt.subplots()
data = [
    df[df["cond"] == "pure"]["final_distance"].values,
    df[df["cond"] == "pi_power"]["final_distance"].values,
]
ax.boxplot(data, tick_labels=[LABELS["pure"], LABELS["pi_power"]])
ax.set_ylabel("Distância final (m)")
fig.tight_layout()
save_fig(fig, "fig_boxplot_final_distance")

# ========= FIG 2: Boxplot duração do episódio (episódios) =========
fig, ax = plt.subplots()
data = [
    df[df["cond"] == "pure"]["length"].values,
    df[df["cond"] == "pi_power"]["length"].values,
]
ax.boxplot(data, tick_labels=[LABELS["pure"], LABELS["pi_power"]])
ax.set_ylabel("Duração do episódio (passos)")
fig.tight_layout()
save_fig(fig, "fig_boxplot_episode_length")

# ========= FIG 3: Sucesso por seed (barra com dispersão implícita via std) =========
# (MUITO bom para slide e para banca)
fig, ax = plt.subplots()
order = ["pure", "pi_power"]
means = [by_seed[by_seed["cond"] == c]["success_rate"].mean() for c in order]
stds = [by_seed[by_seed["cond"] == c]["success_rate"].std(ddof=1) for c in order]

ax.bar([LABELS[c] for c in order], means, yerr=stds)
ax.set_ylabel("Taxa de sucesso (média ± dp entre seeds)")
fig.tight_layout()
save_fig(fig, "fig_success_rate_mean_std_seeds")

# ========= FIG 4: Boxplot esforço em torque por episódio (se existir) =========
if "mean_tau_l1" in df.columns:
    fig, ax = plt.subplots()
    data = [
        df[df["cond"] == "pure"]["mean_tau_l1"].values,
        df[df["cond"] == "pi_power"]["mean_tau_l1"].values,
    ]
    ax.boxplot(data, tick_labels=[LABELS["pure"], LABELS["pi_power"]])
    ax.set_ylabel(r"Esforço médio em torque ($\|\tau\|_1$ por episódio)")
    fig.tight_layout()
    save_fig(fig, "fig_boxplot_tau_l1_effort")

# ========= FIG 5: Boxplot power abs por episódio (se existir) =========
if "mean_power_abs" in df.columns:
    fig, ax = plt.subplots()
    data = [
        df[df["cond"] == "pure"]["mean_power_abs"].values,
        df[df["cond"] == "pi_power"]["mean_power_abs"].values,
    ]
    ax.boxplot(data, tick_labels=[LABELS["pure"], LABELS["pi_power"]])
    ax.set_ylabel(r"Potência média abs (proxy) por episódio")
    fig.tight_layout()
    save_fig(fig, "fig_boxplot_power_abs")

# ========= FIG 6: Histograma lado a lado da distância final =========
bins = 20
range_min = 0.0
range_max = float(max(df["final_distance"].max(), 0.05))
bin_edges = np.linspace(range_min, range_max, bins + 1)

pure = df[df["cond"] == "pure"]["final_distance"].values
pi = df[df["cond"] == "pi_power"]["final_distance"].values

counts_pure, _ = np.histogram(pure, bins=bin_edges)
counts_pi, _ = np.histogram(pi, bins=bin_edges)

centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
width = bin_edges[1] - bin_edges[0]
bar_w = width * 0.4
offset = bar_w

fig, ax = plt.subplots()
ax.bar(centers - offset / 2, counts_pure, width=bar_w, label=LABELS["pure"])
ax.bar(centers + offset / 2, counts_pi, width=bar_w, label=LABELS["pi_power"])
ax.set_xlabel("Distância final (m)")
ax.set_ylabel("Frequência")
ax.legend()
fig.tight_layout()
save_fig(fig, "fig_hist_final_distance_side_by_side")

# ========= FIG 7: Trade-off por seed (desempenho × custo físico) =========
# Se tiver power, usa power. Senão usa torque.
cost_col = None
cost_label = None
if "mean_power_abs" in by_seed.columns:
    cost_col = "mean_power_abs"
    cost_label = "Custo físico (power abs médio por seed)"
elif "mean_tau_l1" in by_seed.columns:
    cost_col = "mean_tau_l1"
    cost_label = r"Custo físico ($\|\tau\|_1$ médio por seed)"

if cost_col is not None:
    fig, ax = plt.subplots()
    for cond, sub in by_seed.groupby("cond"):
        ax.scatter(
            sub[cost_col].values,
            sub["success_rate"].values,
            label=LABELS.get(cond, cond),
        )
    ax.set_xlabel(cost_label)
    ax.set_ylabel("Taxa de sucesso por seed")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_tradeoff_success_vs_cost_by_seed")

print("OK — Figuras em:", OUT_DIR)
print("OK — Tabelas em:", TABELAS_DIR)
