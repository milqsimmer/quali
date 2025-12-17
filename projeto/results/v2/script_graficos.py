# -*- coding: utf-8 -*-
"""
Gera Tabela 1 e Figuras 1–3 e 5 do capítulo de Resultados.
Requisitos: pandas, numpy, matplotlib. (Sem seaborn; 1 gráfico por figura; sem cores definidas.)

Edite apenas:
- CSV_PATH
- OUT_DIR  (opcional)

CSV esperado (colunas):
mode,episode,success,final_distance,return,length
pure,1,0,0.0950,-26.05,200
pirl,1,1,0.0340,-12.80,87
...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES GERAIS =========
CSV_PATH = r"results_torque.csv"
OUT_DIR = r"figs_resultados"
TABELAS_DIR = r"tabelas"

# ========= LABELS =========
LABELS = {
    "pure": "RL puro",
    "baseline": "RL puro",
    "rl": "RL puro",
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

# ========= NORMALIZAÇÃO DE COLUNAS =========
# Renomeia se vierem variantes de nome
rename_map = {
    "final_dist": "final_distance",
    "ep_len": "length",
    "ret": "return",
}
df.rename(
    columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
)

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


# ========= NORMALIZAÇÃO DE 'mode' =========
def norm_mode(x: str) -> str:
    s = str(x).lower()
    if "pirl" in s or s == "pi-rl":
        return "pirl"
    if s in ["pure", "baseline", "rl"]:
        return "pure"
    # fallback: tudo que não for 'pirl' vira 'pure'
    return "pure"


df["mode_norm"] = df["mode"].map(norm_mode)
df["mode_label"] = df["mode_norm"].map(lambda m: LABELS.get(m, m))


# ========= FUNÇÕES AUXILIARES =========
def mean_std(series):
    return series.mean(), series.std(ddof=1)


def resumo_metrica(df_mode):
    succ = df_mode["success"].mean()  # taxa (0..1)
    d_mean, d_std = mean_std(df_mode["final_distance"])
    l_mean, l_std = mean_std(df_mode["length"])
    r_mean, r_std = mean_std(df_mode["return"])
    return succ, d_mean, d_std, l_mean, l_std, r_mean, r_std


# # ========= TABELA 1: MÉDIAS ± DESVIOS =========
# linhas = []
# for modo in ["pure", "pirl"]:
#     sub = df[df["mode_norm"] == modo]
#     if sub.empty:
#         continue
#     succ, d_m, d_s, l_m, l_s, r_m, r_s = resumo_metrica(sub)
#     linhas.append(
#         {
#             "Modo": LABELS.get(modo, modo),
#             "Sucesso": round(succ, 2),
#             "Distância final (m)": f"{d_m:.4f} ± {d_s:.4f}",
#             "Duração (passos)": f"{l_m:.2f} ± {l_s:.2f}",
#             "Retorno": f"{r_m:.2f} ± {r_s:.2f}",
#         }
#     )

# tabela = pd.DataFrame(
#     linhas,
#     columns=["Modo", "Sucesso", "Distância final (m)", "Duração (passos)", "Retorno"],
# )

# # Salva CSV e LaTeX da Tabela 1
# tabela_csv_path = os.path.join(TABELAS_DIR, "tabela1_resultados.csv")
# tabela.to_csv(tabela_csv_path, index=False, encoding="utf-8-sig")

# tabela_tex_path = os.path.join(TABELAS_DIR, "tabela1_resultados.tex")
# with open(tabela_tex_path, "w", encoding="utf-8") as f:
#     f.write(r"\begin{tabular}{lcccc}" + "\n")
#     f.write(r"\hline" + "\n")
#     f.write(
#         r"\textbf{Modo} & \textbf{Sucesso} & \textbf{Distância final (m)} & \textbf{Duração (passos)} & \textbf{Retorno} \\"
#         + "\n"
#     )
#     f.write(r"\hline" + "\n")
#     for _, row in tabela.iterrows():
#         f.write(
#             f"{row['Modo']} & {row['Sucesso']} & {row['Distância final (m)']} & {row['Duração (passos)']} & {row['Retorno']} \\\\\n"
#         )
#     f.write(r"\hline" + "\n")
#     f.write(r"\end{tabular}" + "\n")

# print("Tabela 1 salva em:", tabela_csv_path)
# print("LaTeX salvo em:", tabela_tex_path)

# # ========= FIGURA 1: BOXPlot DISTÂNCIAS FINAIS =========
fig1, ax1 = plt.subplots()
data_pure = df[df["mode_norm"] == "pure"]["final_distance"].values
data_pirl = df[df["mode_norm"] == "pirl"]["final_distance"].values
ax1.boxplot([data_pure, data_pirl], tick_labels=[LABELS["pure"], LABELS["pirl"]])
ax1.set_ylabel("Distância final (m)")
# ax1.set_title("Figura 1 – Distância final por condição (boxplot)")
fig1.tight_layout()
# fig1_path = os.path.join(OUT_DIR, "fig1_boxplot_dist_final.png")
# fig1.savefig(fig1_path, dpi=300)
fig1_path = os.path.join(OUT_DIR, "fig1_boxplot_dist_final.pdf")
plt.savefig(fig1_path, format="pdf", bbox_inches="tight")
plt.close(fig1)

# ========= FIGURA 2: BOXPlot DURAÇÃO =========
fig2, ax2 = plt.subplots()
len_pure = df[df["mode_norm"] == "pure"]["length"].values
len_pirl = df[df["mode_norm"] == "pirl"]["length"].values
ax2.boxplot([len_pure, len_pirl], tick_labels=[LABELS["pure"], LABELS["pirl"]])
ax2.set_ylabel("Duração do episódio (passos)")
# ax2.set_title("Figura 2 – Duração por condição (boxplot)")
fig2.tight_layout()
# fig2_path = os.path.join(OUT_DIR, "fig2_boxplot_duracao.png")
# fig2.savefig(fig2_path, dpi=300)
fig2_path = os.path.join(OUT_DIR, "fig2_boxplot_duracao.pdf")
plt.savefig(fig2_path, format="pdf", bbox_inches="tight")
plt.close(fig2)


# ========= FIGURA 3: TAXA DE SUCESSO ACUMULADA =========
def taxa_acumulada(sucessos):
    sucesso_cum = np.cumsum(sucessos)
    idx = np.arange(1, len(sucessos) + 1)
    return sucesso_cum / idx


fig3, ax3 = plt.subplots()
for modo in ["pure", "pirl"]:
    sub = df[df["mode_norm"] == modo].sort_values("episode")
    y = taxa_acumulada(sub["success"].values.astype(float))
    x = np.arange(1, len(y) + 1)
    ax3.plot(x, y, label=LABELS.get(modo, modo))
ax3.set_xlabel("Episódio de avaliação")
ax3.set_ylabel("Taxa de sucesso acumulada")
# ax3.set_title("Figura 3 – Taxa de sucesso acumulada por condição")
ax3.legend()
fig3.tight_layout()
# fig3_path = os.path.join(OUT_DIR, "fig3_sucesso_acumulado.png")
# fig3.savefig(fig3_path, dpi=300)
fig3_path = os.path.join(OUT_DIR, "fig3_sucesso_acumulado.pdf")
plt.savefig(fig3_path, format="pdf", bbox_inches="tight")
plt.close(fig3)

# ========= FIGURA 5 (lado a lado): Histograma de distâncias finais =========
# 1) calculamos contagens manualmente para cada modo usando as MESMAS bordas de bins
bins = 20  # ajuste se quiser
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
# ax5.set_title("Figura 5 – Histograma de distâncias finais (barras lado a lado)")
ax5.legend()
fig5.tight_layout()

# fig5_path = os.path.join(OUT_DIR, "fig5_hist_dist_final_lado_a_lado.png")
# fig5.savefig(fig5_path, dpi=300)

fig5_path = os.path.join(OUT_DIR, "fig5_hist_dist_final_lado_a_lado.pdf")
plt.savefig(fig5_path, format="pdf", bbox_inches="tight")

plt.close(fig5)

# ========= FIGURA 66: Boxplot Torques =========

# Verifica rapidamente se as colunas esperadas existem
required_cols = {"mode", "mean_tau_l1"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Colunas faltantes no CSV: {missing}. Colunas disponíveis: {df.columns.tolist()}"
    )

# === 2. Separar dados por condição ===
modes = ["pure", "pirl"]
labels = ["RL puro", "PIRL"]

data = []
for m in modes:
    sub = df[df["mode"] == m]["mean_tau_l1"].values
    if len(sub) == 0:
        raise ValueError(f"Nenhum dado encontrado para mode='{m}'. Verifique o CSV.")
    data.append(sub)

# === 3. Criar o boxplot ===
fig, ax = plt.subplots()

ax.boxplot(data, tick_labels=labels)

ax.set_ylabel(r"Esforço médio em torque $E_i$ ($\|\tau\|_1$ médio por episódio)")
# ax.set_title("Distribuição do esforço médio em torque por condição")

fig.tight_layout()

# === 4. Salvar figura ===
fig6_path = os.path.join(OUT_DIR, "boxplot_torque_effort.pdf")
plt.savefig(fig6_path, format="pdf", bbox_inches="tight")
plt.close(fig)
# ========= FIM =========
