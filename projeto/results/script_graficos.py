# -*- coding: utf-8 -*-
"""
Gera Tabela 1 e Figuras 1–3 e 5 do capítulo de Resultados.
Requisitos: pandas, numpy, matplotlib. (Sem seaborn.)

Onde preencher:

{{caminho_para_csv}}, {{caminho_para_salvar}}

{{n_fig1}}, {{n_fig2}}, {{n_fig3}}, {{n_fig5}}

{{numero_de_bins}} (sugestão: 20)


CHECAR SE:
mode,episode,success,final_dist,ep_len,ret
pure,0,0,0.182,200,-41.2
pirl,0,1,0.034,87,-12.8
...
mode ∈ {pure,pirl} (ou rl,pirl — ajuste no filtro se usar outros nomes)

episode = índice do episódio (0..99)

success ∈ {0,1}

final_dist em metros

ep_len = passos até término

ret = retorno do episódio
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIGURAÇÕES GERAIS =========
CSV_PATH = r"eval_results.csv"
OUT_DIR = r"figs_resultados"
TABELAS_DIR = r"tabelas"

# nomes amigáveis (rótulos nas legendas e figuras)
LABELS = {
    "pure": "RL puro",
    "rl": "RL puro",
    "pirl": "PIRL (PI-reward)",
}

# cria pastas
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABELAS_DIR, exist_ok=True)

# ========= CARREGAMENTO =========
df = pd.read_csv(CSV_PATH)

# normaliza nome de modo
df["mode_norm"] = (
    df["mode"].str.lower().map(lambda x: "pirl" if "pirl" in x else "pure")
)
df["mode_label"] = df["mode_norm"].map(LABELS)


# ========= TABELA 1: médias ± desvios =========
def mean_std(series):
    return series.mean(), series.std(ddof=1)


def resumo_metrica(df_mode):
    succ = df_mode["success"].mean()
    d_mean, d_std = mean_std(df_mode["final_distance"])
    l_mean, l_std = mean_std(df_mode["length"])
    r_mean, r_std = mean_std(df_mode["return"])
    return succ, d_mean, d_std, l_mean, l_std, r_mean, r_std


linhas = []
for modo in ["pure", "pirl"]:
    sub = df[df["mode"] == modo]
    if sub.empty:
        continue
    succ, d_m, d_s, l_m, l_s, r_m, r_s = resumo_metrica(sub)
    linhas.append(
        {
            "Modo": LABELS[modo],
            "Sucesso": round(succ, 2),
            "Distância final (m)": f"{d_m:.4f} ± {d_s:.4f}",
            "Duração (passos)": f"{l_m:.2f} ± {l_s:.2f}",
            "Retorno": f"{r_m:.2f} ± {r_s:.2f}",
        }
    )

tabela = pd.DataFrame(
    linhas,
    columns=["Modo", "Sucesso", "Distância final (m)", "Duração (passos)", "Retorno"],
)

# salva versões da Tabela 1
tabela.to_csv(os.path.join(TABELAS_DIR, "tabela1_resultados.csv"), index=False)

# exporta LaTeX simples (cole no seu .tex ou no Word via conversão)
with open(
    os.path.join(TABELAS_DIR, "tabela1_resultados.tex"), "w", encoding="utf-8"
) as f:
    f.write(r"\begin{tabular}{lcccc}" + "\n")
    f.write(r"\hline" + "\n")
    f.write(
        r"\textbf{Modo} & \textbf{Sucesso} & \textbf{Distância final (m)} & \textbf{Duração (passos)} & \textbf{Retorno} \\"
        + "\n"
    )
    f.write(r"\hline" + "\n")
    for _, row in tabela.iterrows():
        f.write(
            f"{row['Modo']} & {row['Sucesso']} & {row['Distância final (m)']} & {row['Duração (passos)']} & {row['Retorno']} \\\\\n"
        )
    f.write(r"\hline" + "\n")
    f.write(r"\end{tabular}" + "\n")

print("Tabela 1 salva em:", os.path.join(TABELAS_DIR, "tabela1_resultados.csv"))

# # ========= FIGURA 1: Boxplot de distâncias finais =========
# # Um gráfico por figura; sem cores definidas.
# fig1, ax1 = plt.subplots()
# data_pure = df[df["mode_norm"] == "pure"]["final_dist"].values
# data_pirl = df[df["mode_norm"] == "pirl"]["final_dist"].values
# ax1.boxplot([data_pure, data_pirl], labels=[LABELS["pure"], LABELS["pirl"]])
# ax1.set_ylabel("Distância final (m)")
# ax1.set_title("Figura {{n_fig1}} – Distância final por condição (boxplot)")
# fig1.tight_layout()
# fig1_path = os.path.join(OUT_DIR, "fig1_boxplot_dist_final.png")
# fig1.savefig(fig1_path, dpi=300)
# plt.close(fig1)

# # ========= FIGURA 2: Boxplot de duração =========
# fig2, ax2 = plt.subplots()
# len_pure = df[df["mode_norm"] == "pure"]["ep_len"].values
# len_pirl = df[df["mode_norm"] == "pirl"]["ep_len"].values
# ax2.boxplot([len_pure, len_pirl], labels=[LABELS["pure"], LABELS["pirl"]])
# ax2.set_ylabel("Duração do episódio (passos)")
# ax2.set_title("Figura {{n_fig2}} – Duração por condição (boxplot)")
# fig2.tight_layout()
# fig2_path = os.path.join(OUT_DIR, "fig2_boxplot_duracao.png")
# fig2.savefig(fig2_path, dpi=300)
# plt.close(fig2)


# # ========= FIGURA 3: Taxa de sucesso acumulada =========
# def taxa_acumulada(sucessos):
#     # retorna série da taxa acumulada episódio a episódio
#     sucesso_cum = np.cumsum(sucessos)
#     idx = np.arange(1, len(sucessos) + 1)
#     return sucesso_cum / idx


# fig3, ax3 = plt.subplots()
# for modo in ["pure", "pirl"]:
#     sub = df[df["mode_norm"] == modo].sort_values("episode")
#     y = taxa_acumulada(sub["success"].values)
#     x = np.arange(1, len(y) + 1)
#     ax3.plot(x, y, label=LABELS[modo])
# ax3.set_xlabel("Episódio de avaliação")
# ax3.set_ylabel("Taxa de sucesso acumulada")
# ax3.set_title("Figura {{n_fig3}} – Taxa de sucesso acumulada por condição")
# ax3.legend()
# fig3.tight_layout()
# fig3_path = os.path.join(OUT_DIR, "fig3_sucesso_acumulado.png")
# fig3.savefig(fig3_path, dpi=300)
# plt.close(fig3)

# # ========= FIGURA 5: Histograma de distâncias finais =========
# # dois histogramas sobrepostos em bins iguais (um gráfico por figura)
# bins = {{numero_de_bins}}  # ex.: 20
# range_min = 0.0
# range_max = float(max(df["final_dist"].max(), 0.05))
# fig5, ax5 = plt.subplots()
# ax5.hist(
#     data_pure, bins=bins, range=(range_min, range_max), alpha=0.5, label=LABELS["pure"]
# )
# ax5.hist(
#     data_pirl, bins=bins, range=(range_min, range_max), alpha=0.5, label=LABELS["pirl"]
# )
# ax5.set_xlabel("Distância final (m)")
# ax5.set_ylabel("Frequência")
# ax5.set_title("Figura {{n_fig5}} – Histograma de distâncias finais")
# ax5.legend()
# fig5.tight_layout()
# fig5_path = os.path.join(OUT_DIR, "fig5_hist_dist_final.png")
# fig5.savefig(fig5_path, dpi=300)
# plt.close(fig5)

# print("Figuras salvas em:", OUT_DIR)
