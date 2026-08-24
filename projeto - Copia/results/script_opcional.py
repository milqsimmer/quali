2) Figura 4 — Trajetórias XY (quando você tiver logs de trajetórias)

Para esta figura, você precisa ter salvo as trajetórias da ponta por episódio, algo como:

traj/
  pure_ep000.npy  -> array shape (T, 2) com [x,y] da ponta
  pure_ep001.npy
  ...
  pirl_ep000.npy
  ...


salve como {{caminho_para_salvar}}/gera_trajetorias.py

# -*- coding: utf-8 -*-
"""
Gera Figura 4 – Trajetórias XY (exemplos) para RL puro e PIRL.
Lê arquivos .npy com shape (T,2) contendo (x,y) da ponta por episódio.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

TRAJ_DIR = r"{{caminho_para_traj}}/traj"
OUT_DIR  = r"{{caminho_para_salvar}}/figs_resultados"
os.makedirs(OUT_DIR, exist_ok=True)

# escolha dos episódios "representativos"
EPISODIOS_PURE = {{lista_eps_pure}}   # ex.: [3, 17, 42]
EPISODIOS_PIRL = {{lista_eps_pirl}}   # ex.: [5, 21, 37]

def carrega_traj(mode, ep):
    fname = f"{mode}_ep{ep:03d}.npy"
    path = os.path.join(TRAJ_DIR, fname)
    arr = np.load(path)  # shape (T,2)
    return arr

# --------- Figura 4a: RL puro ----------
fig4a, ax4a = plt.subplots()
for ep in EPISODIOS_PURE:
    xy = carrega_traj("pure", ep)
    ax4a.plot(xy[:,0], xy[:,1], linewidth=1)
ax4a.set_aspect("equal", adjustable="box")
ax4a.set_xlabel("x (m)")
ax4a.set_ylabel("y (m)")
ax4a.set_title("Figura {{n_fig4a}} – Trajetórias XY (RL puro)")
fig4a.tight_layout()
fig4a_path = os.path.join(OUT_DIR, "fig4a_traj_xy_pure.png")
fig4a.savefig(fig4a_path, dpi=300)
plt.close(fig4a)

# --------- Figura 4b: PIRL ----------
fig4b, ax4b = plt.subplots()
for ep in EPISODIOS_PIRL:
    xy = carrega_traj("pirl", ep)
    ax4b.plot(xy[:,0], xy[:,1], linewidth=1)
ax4b.set_aspect("equal", adjustable="box")
ax4b.set_xlabel("x (m)")
ax4b.set_ylabel("y (m)")
ax4b.set_title("Figura {{n_fig4b}} – Trajetórias XY (PIRL)")
fig4b.tight_layout()
fig4b_path = os.path.join(OUT_DIR, "fig4b_traj_xy_pirl.png")
fig4b.savefig(fig4b_path, dpi=300)
plt.close(fig4b)

print("Figuras de trajetórias salvas em:", OUT_DIR)


Onde preencher:

{{caminho_para_traj}}, {{caminho_para_salvar}}

{{lista_eps_pure}}, {{lista_eps_pirl}} (use episódios que visualmente mostrem trajetória estável/eficiente)

{{n_fig4a}}, {{n_fig4b}}

Dica: se você ainda não salva as trajetórias, adicione no loop de avaliação algo como:

# ao final de cada episódio:
np.save(f"{save_dir}/{mode}_ep{ep:03d}.npy", np.array(traj_xy))  # traj_xy: lista de [x,y] por step
