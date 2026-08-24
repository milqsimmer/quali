import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_GLOB = r"results/eval_official_seed*_*.csv"
OUT_DIR = "figs_resultados"
TABELAS_DIR = "tabelas"

LABELS = {
    "pure": "RL puro",
    "pirl": "PIRL (PI-reward)",
}


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def normalize_mode(value) -> str:
    if pd.isna(value):
        return ""
    v = str(value).strip().lower()
    if "pirl" in v or "pi-reward" in v or "physics" in v:
        return "pirl"
    if v in {"pure", "rl", "baseline"} or "pure" in v or "baseline" in v:
        return "pure"
    return v


def main() -> None:
    import glob

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TABELAS_DIR, exist_ok=True)

    paths = sorted(glob.glob(CSV_GLOB))
    if not paths:
        raise SystemExit(f"Nenhum CSV encontrado com padrão: {CSV_GLOB}")

    dfs = [read_csv_auto(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    required = ["mode", "success", "final_distance", "tip_init_dist"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas faltando nos CSVs: {missing}")

    df["mode_norm"] = df["mode"].apply(normalize_mode)
    df = df[df["mode_norm"].isin(["pure", "pirl"])].copy()

    num_cols = ["success", "final_distance", "tip_init_dist"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["mode_norm", "success", "final_distance", "tip_init_dist"])

    # define bins de distancia inicial da ponta ao alvo (em metros)
    # ajuste se necessário dependendo da distribuição dos alvos
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]

    df["radius_bin"] = pd.cut(df["tip_init_dist"], bins=bins, labels=bin_labels, include_lowest=True)
    df = df.dropna(subset=["radius_bin"])  # remove alvos fora dos bins

    # agrega por modo e bin
    grp = (
        df.groupby(["mode_norm", "radius_bin"], observed=True)
        .agg(
            success_rate=("success", "mean"),
            final_distance=("final_distance", "mean"),
            n_episodes=("success", "count"),
        )
        .reset_index()
        .sort_values(["radius_bin", "mode_norm"])
    )

    grp["mode_label"] = grp["mode_norm"].map(LABELS)
    out_csv = os.path.join(TABELAS_DIR, "target_distance_success.csv")
    grp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # grafico: taxa de sucesso vs raio inicial
    fig, ax = plt.subplots()

    x = np.arange(len(bin_labels))
    width = 0.35

    data_pure = grp[grp["mode_norm"] == "pure"].set_index("radius_bin")["success_rate"]
    data_pirl = grp[grp["mode_norm"] == "pirl"].set_index("radius_bin")["success_rate"]

    # garante que todos os bins estejam presentes
    pure_vals = [float(data_pure.get(lbl, np.nan)) for lbl in bin_labels]
    pirl_vals = [float(data_pirl.get(lbl, np.nan)) for lbl in bin_labels]

    ax.bar(x - width / 2, pure_vals, width, label=LABELS["pure"])
    ax.bar(x + width / 2, pirl_vals, width, label=LABELS["pirl"])

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Taxa de sucesso")
    ax.set_xlabel("Distância inicial da ponta ao alvo (m)")
    ax.legend()
    fig.tight_layout()

    out_fig = os.path.join(OUT_DIR, "fig_target_distance_success.pdf")
    plt.savefig(out_fig, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Resumo salvo em:", out_csv)
    print("Figura salva em:", out_fig)


if __name__ == "__main__":
    main()
