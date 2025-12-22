# plot_results.py

"""
Como usar na dissertação:

agg_by_model.csv vira tabela (success rate, dist final média, esforço médio de torque, etc.).

success_by_d0bins_*.png vira uma figura: robustez vs dificuldade (d0).

Boxplots por modo ajudam a comparar distribuição (pure vs pirl).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_success_by_model(agg_by_model_csv: str, out_dir: str):
    df = pd.read_csv(agg_by_model_csv)
    # ordena para ficar consistente
    df = df.sort_values(["mode", "train_seed"])

    tick_labels = [f"{m}-seed{int(s)}" for m, s in zip(df["mode"], df["train_seed"])]
    y = df["success_rate"].astype(float).values

    plt.figure()
    plt.bar(np.arange(len(y)), y)
    plt.xticks(np.arange(len(y)), tick_labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.title("Success rate por modelo (mode × train_seed)")
    _savefig(os.path.join(out_dir, "success_rate_by_model.png"))


def plot_metric_box(merged_csv: str, out_dir: str, metric: str, title: str):
    df = pd.read_csv(merged_csv)
    df = df.dropna(subset=[metric, "mode"])
    modes = sorted(df["mode"].unique())

    data = [df[df["mode"] == m][metric].astype(float).values for m in modes]

    plt.figure()
    plt.boxplot(data, tick_labels=modes)
    plt.ylabel(metric)
    plt.title(title)
    _savefig(os.path.join(out_dir, f"box_{metric}_by_mode.png"))


def plot_success_by_d0_bins(agg_bins_csv: str, out_dir: str):
    df = pd.read_csv(agg_bins_csv)

    # d0_bin vem como string tipo "(0.07, 0.3]"
    # vamos plotar por modo separadamente
    modes = sorted(df["mode"].dropna().unique())

    for m in modes:
        sub = df[df["mode"] == m].copy()

        # ordena bins por limite inferior (parse simples)
        def bin_left(s):
            try:
                s = str(s)
                left = s.split(",")[0].replace("(", "").replace("[", "").strip()
                return float(left)
            except:
                return 0.0

        sub["bin_left"] = sub["d0_bin"].apply(bin_left)
        sub = sub.sort_values(["train_seed", "bin_left"])

        # uma figura por train_seed para ficar legível
        for seed in sorted(sub["train_seed"].dropna().unique()):
            sub2 = sub[sub["train_seed"] == seed]
            x = np.arange(len(sub2))
            y = sub2["success_rate"].astype(float).values
            tick_labels = sub2["d0_bin"].astype(str).values

            plt.figure()
            plt.plot(x, y, marker="o")
            plt.xticks(x, tick_labels, rotation=45, ha="right")
            plt.ylim(0, 1)
            plt.ylabel("Success rate")
            plt.title(f"Success rate por bins de d0 — mode={m}, train_seed={int(seed)}")
            _savefig(
                os.path.join(out_dir, f"success_by_d0bins_{m}_seed{int(seed)}.png")
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="results_agg/merged_eval.csv")
    ap.add_argument("--agg-model", default="results_agg/agg_by_model.csv")
    ap.add_argument("--agg-bins", default="results_agg/agg_by_d0_bins.csv")
    ap.add_argument("--out-dir", default="results_agg/figs")
    args = ap.parse_args()

    plot_success_by_model(args.agg_model, args.out_dir)

    # Boxplots principais (por modo)
    plot_metric_box(
        args.merged, args.out_dir, "final_distance", "Distância final por modo"
    )
    plot_metric_box(
        args.merged, args.out_dir, "length", "Comprimento do episódio por modo"
    )
    plot_metric_box(
        args.merged, args.out_dir, "mean_tau_l1", "Esforço de torque médio por modo"
    )

    # Componentes (se existirem)
    # (não falha se estiverem ausentes, apenas não plotar)
    merged_df = pd.read_csv(args.merged)
    for metric, title in [
        ("sum_r_dist", "Soma de r_dist por episódio (por modo)"),
        ("sum_r_act", "Soma de r_act por episódio (por modo)"),
        ("sum_r_tau", "Soma de r_tau por episódio (por modo)"),
        ("mean_r_tau", "Média de r_tau por passo (por modo)"),
    ]:
        if metric in merged_df.columns:
            plot_metric_box(args.merged, args.out_dir, metric, title)

    plot_success_by_d0_bins(args.agg_bins, args.out_dir)

    print("[OK] Figuras geradas em:", args.out_dir)


if __name__ == "__main__":
    main()
