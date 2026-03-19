import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES GERAIS =========
CSV_PATH = r"eval_results_0.csv"
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
        # tenta separador ';' (CSV padrão BR)
        return pd.read_csv(path, sep=";")


df = read_csv_auto(CSV_PATH)

# Checagem mínima
required_cols = ["mode", "episode", "success", "final_distance", "return", "length"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Colunas faltando no CSV: {missing}. Precisam ser {required_cols}"
    )

# ========= TIPOS NUMÉRICOS =========
num_cols = ["episode", "success", "final_distance", "return", "length"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Remove linhas com NaN em colunas essenciais
df = df.dropna(subset=["mode", "success", "final_distance", "return", "length"])


# ========= FUNÇÕES AUXILIARES =========
def mean_std(series):
    return series.mean(), series.std(ddof=1)


def resumo_metrica(df_mode):
    succ = df_mode["success"].mean()  # taxa (0..1)
    d_mean, d_std = mean_std(df_mode["final_distance"])
    l_mean, l_std = mean_std(df_mode["length"])
    r_mean, r_std = mean_std(df_mode["return"])
    return succ, d_mean, d_std, l_mean, l_std, r_mean, r_std


# # ========= FIGURA 1: BOXPlot DISTÂNCIAS FINAIS =========
fig1, ax1 = plt.subplots()
data_pure = df[df["mode_norm"] == "pure"]["final_distance"].values
data_pirl = df[df["mode_norm"] == "pirl"]["final_distance"].values
ax1.boxplot([data_pure, data_pirl], tick_labels=[LABELS["pure"], LABELS["pirl"]])
ax1.set_ylabel("Distância final (m)")
fig1.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig1_boxplot_dist_final.pdf")
plt.savefig(fig1_path, format="pdf", bbox_inches="tight")
plt.close(fig1)

# ========= FIGURA 2: BOXPlot DURAÇÃO =========
fig2, ax2 = plt.subplots()
len_pure = df[df["mode_norm"] == "pure"]["length"].values
len_pirl = df[df["mode_norm"] == "pirl"]["length"].values
ax2.boxplot([len_pure, len_pirl], tick_labels=[LABELS["pure"], LABELS["pirl"]])
ax2.set_ylabel("Duração do episódio (passos)")
fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig2_boxplot_duracao.pdf")
plt.savefig(fig2_path, format="pdf", bbox_inches="tight")
plt.close(fig2)


# ========= FIGURA 5 (lado a lado): Histograma de distâncias finais =========
# 1) calculamos contagens manualmente para cada modo usando as MESMAS bordas de bins
bins = 20
range_min = 0.0
range_max = float(max(df["final_distance"].max(), 0.05))

# bordas comuns dos bins
bin_edges = np.linspace(range_min, range_max, bins + 1)

# dados por modo
data_pure = df[df["mode_norm"] == "pure"]["final_distance"].values
data_pirl = df[df["mode_norm"] == "pirl"]["final_distance"].values

# contagens por bin (usando as mesmas bordas)
counts_pure, _ = np.histogram(data_pure, bins=bin_edges)
counts_pirl, _ = np.histogram(data_pirl, bins=bin_edges)

# posições no eixo x (centros dos bins) e largura
centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
width = bin_edges[1] - bin_edges[0]

# para ficar lado a lado, deslocamos cada barra metade da largura*0.8
bar_w = width * 0.4  # cada barra ocupará 40% do bin
offset = bar_w

fig5, ax5 = plt.subplots()

# barras lado a lado (sem sobreposição, sem especificar cores)
ax5.bar(centers - offset / 2, counts_pure, width=bar_w, label=LABELS["pure"])
ax5.bar(centers + offset / 2, counts_pirl, width=bar_w, label=LABELS["pirl"])

ax5.set_xlabel("Distância final (m)")
ax5.set_ylabel("Frequência")
ax5.legend()
fig5.tight_layout()
fig5_path = os.path.join(OUT_DIR, "fig5_hist_dist_final_lado_a_lado.pdf")
plt.savefig(fig5_path, format="pdf", bbox_inches="tight")

plt.close(fig5)
