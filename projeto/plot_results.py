# plot_results.py
# Gera figuras a partir dos outputs do aggregate_results.py (setup atual).
#
# Entrada típica:
#   results_agg/aggregated_across_seeds.csv
#   results_agg/aggregated_d0bin_across_seeds.csv
#
# Uso:
#   python plot_results.py --across results_agg/aggregated_across_seeds.csv --d0bins results_agg/aggregated_d0bin_across_seeds.csv --out-dir results_agg/figs
#
# Observação: NÃO misture direct e residual no mesmo "baseline".
# Este script usa 'control' como eixo (coluna) e deixa N/A como NaN.

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AXIS_COLS = ["control", "use_pi_reward", "pi_metric", "alpha_pi", "safety_filter"]


def _savefig(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _make_label(row: pd.Series) -> str:
    parts = [str(row.get("exp_id", ""))]
    for c in AXIS_COLS:
        if c in row and pd.notna(row[c]):
            parts.append(f"{c}={row[c]}")
    return " | ".join(parts)


def _bar_with_ci(df: pd.DataFrame, value_col: str, ci_col: str, title: str, ylabel: str, out_path: str, top_k: int = 30):
    sub = df.copy()
    if value_col not in sub.columns:
        print(f"[skip] coluna ausente: {value_col}")
        return
    sub = sub.sort_values(value_col, ascending=False)
    sub = sub.head(top_k)

    x = np.arange(len(sub))
    y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)

    yerr = None
    if ci_col in sub.columns:
        yerr = pd.to_numeric(sub[ci_col], errors="coerce").to_numpy(dtype=float)

    labels = [_make_label(r) for _, r in sub.iterrows()]

    plt.figure(figsize=(max(8, len(sub) * 0.35), 4))
    if yerr is not None:
        plt.bar(x, y, yerr=yerr)
    else:
        plt.bar(x, y)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig(out_path)


def plot_across_overview(across_csv: str, out_dir: str):
    df = pd.read_csv(across_csv)

    # success rate
    _bar_with_ci(
        df,
        value_col="success_rate__mean",
        ci_col="success_rate__ci95",
        title="Success rate (média sobre train_seeds) — top configs",
        ylabel="success rate",
        out_path=os.path.join(out_dir, "success_rate_across_seeds.png"),
    )

    # final distance (menor é melhor) -> ordenar invertendo sinal
    if "mean_final_distance__mean" in df.columns:
        tmp = df.copy()
        tmp["_neg"] = -pd.to_numeric(tmp["mean_final_distance__mean"], errors="coerce")
        tmp = tmp.sort_values("_neg", ascending=False).drop(columns=["_neg"])
        _bar_with_ci(
            tmp,
            value_col="mean_final_distance__mean",
            ci_col="mean_final_distance__ci95",
            title="Distância final média (média sobre train_seeds) — top configs (menor é melhor)",
            ylabel="final distance",
            out_path=os.path.join(out_dir, "final_distance_across_seeds.png"),
        )

    # esforço tau_l1 (se existir)
    if "mean_tau_l1__mean_ep__mean" in df.columns:
        _bar_with_ci(
            df,
            value_col="mean_tau_l1__mean_ep__mean",
            ci_col="mean_tau_l1__mean_ep__ci95",
            title="Esforço tau_l1 (média por episódio; média sobre train_seeds) — top configs",
            ylabel="mean tau_l1",
            out_path=os.path.join(out_dir, "mean_tau_l1_across_seeds.png"),
        )

    # pi_value (se existir)
    if "mean_pi_value__mean_ep__mean" in df.columns:
        _bar_with_ci(
            df,
            value_col="mean_pi_value__mean_ep__mean",
            ci_col="mean_pi_value__mean_ep__ci95",
            title="PI metric (mean_pi_value) — top configs",
            ylabel="mean pi_value",
            out_path=os.path.join(out_dir, "mean_pi_value_across_seeds.png"),
        )

    # intervenção do filtro (se existir)
    if "filter_intervention_rate__mean_ep__mean" in df.columns:
        _bar_with_ci(
            df,
            value_col="filter_intervention_rate__mean_ep__mean",
            ci_col="filter_intervention_rate__mean_ep__ci95",
            title="Taxa de intervenção do safety filter — top configs",
            ylabel="intervention rate",
            out_path=os.path.join(out_dir, "filter_intervention_rate_across_seeds.png"),
        )


def plot_success_by_d0_bins(d0bins_csv: str, out_dir: str, max_configs: int = 12):
    df = pd.read_csv(d0bins_csv)

    if "success_rate__mean" not in df.columns:
        print("[skip] d0bins sem success_rate__mean")
        return

    # escolhe configs mais relevantes (maior sucesso global médio)
    # (se existir config_id, usa; senão monta label)
    if "config_id" not in df.columns:
        df["config_id"] = df.apply(lambda r: _make_label(r), axis=1)

    # calcula ranking por config no agregado de todos os bins (mean simples)
    rank = (
        df.groupby("config_id")["success_rate__mean"]
        .mean()
        .sort_values(ascending=False)
        .head(max_configs)
        .index.tolist()
    )
    df = df[df["config_id"].isin(rank)].copy()

    # ordena bins pelo limite inferior
    def bin_left(s):
        try:
            s = str(s)
            left = s.split(",")[0].replace("(", "").replace("[", "").strip()
            return float(left)
        except Exception:
            return 0.0

    df["bin_left"] = df["d0_bin"].apply(bin_left)
    df = df.sort_values(["config_id", "bin_left"])

    # uma figura com várias curvas (até max_configs)
    plt.figure(figsize=(10, 5))
    for cfg, sub in df.groupby("config_id", dropna=False):
        x = np.arange(len(sub))
        y = pd.to_numeric(sub["success_rate__mean"], errors="coerce").to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=str(cfg))

    # ticks conforme bins (assume bins iguais entre configs)
    bins_sorted = df.sort_values("bin_left")["d0_bin"].astype(str).unique().tolist()
    plt.xticks(np.arange(len(bins_sorted)), bins_sorted, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("success rate")
    plt.title("Success rate por bins de dificuldade (d0) — top configs")
    plt.legend(fontsize=7, loc="best")
    _savefig(os.path.join(out_dir, "success_by_d0bins_top_configs.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--across", default="results_agg/aggregated_across_seeds.csv")
    ap.add_argument("--d0bins", default="results_agg/aggregated_d0bin_across_seeds.csv")
    ap.add_argument("--out-dir", default="results_agg/figs")
    ap.add_argument("--max-configs", type=int, default=12, help="máx. curvas no plot por d0 bins")
    args = ap.parse_args()

    plot_across_overview(args.across, args.out_dir)
    plot_success_by_d0_bins(args.d0bins, args.out_dir, max_configs=args.max_configs)

    print("[OK] Figuras geradas em:", args.out_dir)


if __name__ == "__main__":
    main()
