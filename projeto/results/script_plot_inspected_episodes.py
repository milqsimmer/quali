import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INSPECT_DIR = "results/episodios_inspecao"
OUT_DIR_SINGLE = "figs_episodios_inspecao/single"
OUT_DIR_PAIRED = "figs_episodios_inspecao/paired"

LABELS = {
    "pure": "RL puro",
    "pirl": "PIRL (PI-reward)",
}


# Espaço de trabalho aproximado do braço (disco de raio l1+l2)
# No env atual, l1 = 0.5, l2 = 0.5 -> raio ~ 1.0 m
WORKSPACE_RADIUS = 1.0
WORKSPACE_MARGIN = 0.1


def set_workspace_limits(ax: plt.Axes) -> None:
    """Define limites fixos para o espaço de trabalho do braço.

    Isso evita o "zoom automático" apenas na trajetória da ponta
    e facilita comparar diferentes episódios entre si.
    """
    r = WORKSPACE_RADIUS
    m = WORKSPACE_MARGIN
    ax.set_xlim(-m, r + m)
    ax.set_ylim(-r - m, r + m)
    ax.set_aspect("equal", adjustable="box")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR_SINGLE, exist_ok=True)
    os.makedirs(OUT_DIR_PAIRED, exist_ok=True)


def parse_filename(path: str) -> Tuple[str, int, str]:
    """Extrai (mode, seed, tag) do nome do arquivo.

    Espera padrão: mode_<mode>_seed<seed>_<tag>.csv
    Ex.: mode_pure_seed2_low_energy_success.csv -> ("pure", 2, "low_energy_success")
    """

    name = os.path.basename(path)
    m = re.match(r"mode_(pure|pirl)_seed(\d+)_(.+)\.csv$", name)
    if not m:
        raise ValueError(f"Nome de arquivo nao segue o padrao esperado: {name}")
    mode = m.group(1)
    seed = int(m.group(2))
    tag = m.group(3)
    return mode, seed, tag


def plot_single_episode(df: pd.DataFrame, mode: str, seed: int, tag: str) -> None:
    """Gera gráficos básicos para um único episódio inspecionado."""

    step = df["step"].values
    dist = df["distance"].values
    tau_sum = df["tau_sum"].values
    cum_energy = df["cumulative_energy"].values

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(step, dist)
    axes[0].set_ylabel("Distância ao alvo (m)")

    axes[1].plot(step, tau_sum)
    axes[1].set_ylabel(r"Torque total $|\tau_1|+|\tau_2|$")

    axes[2].plot(step, cum_energy)
    axes[2].set_ylabel("Energia acumulada (unid.)")
    axes[2].set_xlabel("Passo do episódio")

    title_mode = LABELS.get(mode, mode)
    fig.suptitle(f"{title_mode} – seed={seed}, {tag}")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    fname = f"mode_{mode}_seed{seed}_{tag}.pdf"
    out_path = os.path.join(OUT_DIR_SINGLE, fname)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Trajetória da ponta no plano (tip_x, tip_y), se disponível
    if "tip_x" in df.columns and "tip_y" in df.columns:
        tip_x = df["tip_x"].values
        tip_y = df["tip_y"].values

        # alvo (assumindo constante ao longo do episodio)
        target_x = df["target_x"].values[0]
        target_y = df["target_y"].values[0]

        # indices validos (sem NaN)
        valid = np.isfinite(tip_x) & np.isfinite(tip_y)
        if valid.any():
            start_idx = int(np.argmax(valid))
            end_idx = int(len(valid) - 1 - np.argmax(valid[::-1]))
        else:
            start_idx = end_idx = 0

        fig2, ax2 = plt.subplots()

        # trajetoria completa
        ax2.plot(tip_x, tip_y, "-", label="trajetória da ponta")

        # inicio e fim
        ax2.scatter(
            tip_x[start_idx],
            tip_y[start_idx],
            marker="x",
            color="orange",
            label="início",
        )
        ax2.scatter(
            tip_x[end_idx],
            tip_y[end_idx],
            marker="x",
            color="green",
            label="fim",
        )

        # alvo
        ax2.scatter(target_x, target_y, marker="*", color="red", label="alvo")

        ax2.set_xlabel("Tip x (m)")
        ax2.set_ylabel("Tip y (m)")
        set_workspace_limits(ax2)
        ax2.legend()
        ax2.set_title(f"Trajetória da ponta – {title_mode}, seed={seed}, {tag}")
        fig2.tight_layout()

        fname2 = f"mode_{mode}_seed{seed}_{tag}_tip_traj.pdf"
        out_path2 = os.path.join(OUT_DIR_SINGLE, fname2)
        plt.savefig(out_path2, format="pdf", bbox_inches="tight")
        plt.close(fig2)


def plot_paired_episodes(
    df_pure: pd.DataFrame, df_pirl: pd.DataFrame, seed: int, tag: str
) -> None:
    """Compara pure vs pirl para o mesmo seed/tag em três curvas."""

    step_pure = df_pure["step"].values
    step_pirl = df_pirl["step"].values

    dist_pure = df_pure["distance"].values
    dist_pirl = df_pirl["distance"].values

    tau_pure = df_pure["tau_sum"].values
    tau_pirl = df_pirl["tau_sum"].values

    energy_pure = df_pure["cumulative_energy"].values
    energy_pirl = df_pirl["cumulative_energy"].values

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=False)

    # Distância
    axes[0].plot(step_pure, dist_pure, label=LABELS.get("pure", "pure"))
    axes[0].plot(step_pirl, dist_pirl, label=LABELS.get("pirl", "pirl"))
    axes[0].set_ylabel("Distância ao alvo (m)")
    axes[0].legend()

    # Torque
    axes[1].plot(step_pure, tau_pure, label=LABELS.get("pure", "pure"))
    axes[1].plot(step_pirl, tau_pirl, label=LABELS.get("pirl", "pirl"))
    axes[1].set_ylabel(r"Torque total $|\tau_1|+|\tau_2|$")

    # Energia acumulada
    axes[2].plot(step_pure, energy_pure, label=LABELS.get("pure", "pure"))
    axes[2].plot(step_pirl, energy_pirl, label=LABELS.get("pirl", "pirl"))
    axes[2].set_ylabel("Energia acumulada (unid.)")
    axes[2].set_xlabel("Passo do episódio")

    title = f"Comparação por episódio – seed={seed}, {tag}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    fname = f"seed{seed}_{tag}_paired.pdf"
    out_path = os.path.join(OUT_DIR_PAIRED, fname)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Trajetória da ponta (pure vs pirl) se disponível
    if "tip_x" in df_pure.columns and "tip_x" in df_pirl.columns:
        tipx_pure = df_pure["tip_x"].values
        tipy_pure = df_pure["tip_y"].values
        tipx_pirl = df_pirl["tip_x"].values
        tipy_pirl = df_pirl["tip_y"].values

        # alvo (assumindo igual para pure/pirl e constante ao longo do episodio)
        target_x = df_pure["target_x"].values[0]
        target_y = df_pure["target_y"].values[0]

        # indices validos
        valid_pure = np.isfinite(tipx_pure) & np.isfinite(tipy_pure)
        valid_pirl = np.isfinite(tipx_pirl) & np.isfinite(tipy_pirl)

        def start_end_idx(valid_arr: np.ndarray) -> tuple[int, int]:
            if valid_arr.any():
                si = int(np.argmax(valid_arr))
                ei = int(len(valid_arr) - 1 - np.argmax(valid_arr[::-1]))
            else:
                si = ei = 0
            return si, ei

        s_pure, e_pure = start_end_idx(valid_pure)
        s_pirl, e_pirl = start_end_idx(valid_pirl)

        fig2, ax2 = plt.subplots()

        # trajetorias
        ax2.plot(tipx_pure, tipy_pure, "-", label=LABELS.get("pure", "pure"))
        ax2.plot(tipx_pirl, tipy_pirl, "-", label=LABELS.get("pirl", "pirl"))

        # inicios
        ax2.scatter(
            tipx_pure[s_pure],
            tipy_pure[s_pure],
            marker="o",
            color="green",
            label="início (pure)",
        )
        ax2.scatter(
            tipx_pirl[s_pirl],
            tipy_pirl[s_pirl],
            marker="o",
            color="darkgreen",
            label="início (pirl)",
        )

        # fins
        ax2.scatter(
            tipx_pure[e_pure],
            tipy_pure[e_pure],
            marker="*",
            color="orange",
            label="fim (pure)",
        )
        ax2.scatter(
            tipx_pirl[e_pirl],
            tipy_pirl[e_pirl],
            marker="*",
            color="red",
            label="fim (pirl)",
        )

        # alvo
        ax2.scatter(target_x, target_y, marker="x", color="black", label="alvo")

        ax2.set_xlabel("Tip x (m)")
        ax2.set_ylabel("Tip y (m)")
        set_workspace_limits(ax2)
        ax2.legend()
        ax2.set_title(f"Trajetória da ponta – seed={seed}, {tag}")
        fig2.tight_layout()

        fname2 = f"seed{seed}_{tag}_tip_traj_paired.pdf"
        out_path2 = os.path.join(OUT_DIR_PAIRED, fname2)
        plt.savefig(out_path2, format="pdf", bbox_inches="tight")
        plt.close(fig2)


def main() -> None:
    ensure_dirs()

    if not os.path.isdir(INSPECT_DIR):
        raise SystemExit(f"Diretório de inspeção não encontrado: {INSPECT_DIR}")

    files = [
        os.path.join(INSPECT_DIR, f)
        for f in os.listdir(INSPECT_DIR)
        if f.endswith(".csv")
    ]
    if not files:
        raise SystemExit(f"Nenhum CSV encontrado em {INSPECT_DIR}")

    # 1) Plots individuais por arquivo
    for path in files:
        mode, seed, tag = parse_filename(path)
        df = pd.read_csv(path)
        print(f"Gerando gráficos single para {path}...")
        plot_single_episode(df, mode, seed, tag)

    # 2) Plots pareados pure vs pirl por (seed, tag)
    grouped: Dict[Tuple[int, str], Dict[str, str]] = {}
    for path in files:
        mode, seed, tag = parse_filename(path)
        key = (seed, tag)
        grouped.setdefault(key, {})[mode] = path

    for (seed, tag), paths in grouped.items():
        if "pure" in paths and "pirl" in paths:
            print(f"Gerando gráfico pareado para seed={seed}, tag={tag}...")
            df_pure = pd.read_csv(paths["pure"])
            df_pirl = pd.read_csv(paths["pirl"])
            plot_paired_episodes(df_pure, df_pirl, seed, tag)

    print("\nFiguras de episódios individuais em:", OUT_DIR_SINGLE)
    print("Figuras comparando pure vs pirl em:", OUT_DIR_PAIRED)


if __name__ == "__main__":
    main()
