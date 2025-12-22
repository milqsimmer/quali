# merge_results.py
import argparse
import glob
import os
import json
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "mode",
    "train_seed",
    "episode",
    "episode_seed",
    "success",
    "d0",
    "final_distance",
    "return",
    "length",
    "mean_tau_l1",
]

OPTIONAL_COLS = [
    "sum_r_dist",
    "sum_r_act",
    "sum_r_tau",
    "mean_r_dist",
    "mean_r_act",
    "mean_r_tau",
    "tau_l1_sum",
    "eval_seed_base",
    "margin",
    "min_tip_dist",
    "phi_min",
    "phi_max",
    "max_steps",
    "success_tol",
]


def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = os.path.basename(path)
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # garante presença das colunas mínimas (se faltar, cria como NaN)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    # agrega por modo e seed de treino
    grp = df.groupby(["mode", "train_seed"], dropna=False)

    def agg(series):
        return pd.Series(
            {
                "n_episodes": int(series.shape[0]),
            }
        )

    out = grp.apply(
        lambda g: pd.Series(
            {
                "n_episodes": int(len(g)),
                "success_rate": float(np.mean(g["success"].astype(float))),
                "mean_d0": float(np.nanmean(g["d0"].astype(float))),
                "std_d0": float(np.nanstd(g["d0"].astype(float))),
                "mean_final_distance": float(
                    np.nanmean(g["final_distance"].astype(float))
                ),
                "std_final_distance": float(
                    np.nanstd(g["final_distance"].astype(float))
                ),
                "mean_return": float(np.nanmean(g["return"].astype(float))),
                "std_return": float(np.nanstd(g["return"].astype(float))),
                "mean_length": float(np.nanmean(g["length"].astype(float))),
                "std_length": float(np.nanstd(g["length"].astype(float))),
                "mean_tau_effort": float(np.nanmean(g["mean_tau_l1"].astype(float))),
                "std_tau_effort": float(np.nanstd(g["mean_tau_l1"].astype(float))),
                # componentes (se existirem)
                "mean_sum_r_dist": float(np.nanmean(g["sum_r_dist"].astype(float))),
                "mean_sum_r_act": float(np.nanmean(g["sum_r_act"].astype(float))),
                "mean_sum_r_tau": float(np.nanmean(g["sum_r_tau"].astype(float))),
                "mean_mean_r_dist": float(np.nanmean(g["mean_r_dist"].astype(float))),
                "mean_mean_r_act": float(np.nanmean(g["mean_r_act"].astype(float))),
                "mean_mean_r_tau": float(np.nanmean(g["mean_r_tau"].astype(float))),
            }
        )
    ).reset_index()

    return out


def _d0_bins(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    # cria bins fixos (comparável entre rodadas)
    d0 = df["d0"].astype(float)
    df = df.copy()
    df["d0_bin"] = pd.cut(d0, bins=bins, include_lowest=True, right=True)

    grp = df.groupby(["mode", "train_seed", "d0_bin"], dropna=False)

    out = grp.apply(
        lambda g: pd.Series(
            {
                "n": int(len(g)),
                "success_rate": float(np.mean(g["success"].astype(float))),
                "mean_final_distance": float(
                    np.nanmean(g["final_distance"].astype(float))
                ),
                "mean_length": float(np.nanmean(g["length"].astype(float))),
                "mean_tau_effort": float(np.nanmean(g["mean_tau_l1"].astype(float))),
            }
        )
    ).reset_index()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern", default="results/*.csv", help="glob dos CSVs do eval (A3)"
    )
    ap.add_argument("--out-dir", default="results_agg", help="pasta de saída")
    ap.add_argument(
        "--d0-bins",
        default="0.07,0.30,0.50,0.70,0.90,1.10,1.40",
        help="bins fixos para d0 (separados por vírgula)",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"Nenhum arquivo encontrado no pattern: {args.pattern}")

    dfs = []
    for p in paths:
        df = _safe_read_csv(p)
        df = _ensure_columns(df)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    merged_path = os.path.join(args.out_dir, "merged_eval.csv")
    merged.to_csv(merged_path, index=False)

    agg_model = _aggregate_by_model(merged)
    agg_model_path = os.path.join(args.out_dir, "agg_by_model.csv")
    agg_model.to_csv(agg_model_path, index=False)

    bins = [float(x.strip()) for x in args.d0_bins.split(",") if x.strip()]
    if len(bins) < 3:
        raise SystemExit(
            "Forneça pelo menos 3 valores em --d0-bins (ex.: 0.07,0.5,1.4)."
        )

    agg_bins = _d0_bins(merged, bins=bins)
    agg_bins_path = os.path.join(args.out_dir, "agg_by_d0_bins.csv")
    agg_bins.to_csv(agg_bins_path, index=False)

    summary = {
        "input_pattern": args.pattern,
        "n_files": len(paths),
        "files": [os.path.basename(p) for p in paths],
        "rows_total": int(len(merged)),
        "d0_bins": bins,
        "outputs": {
            "merged_eval": merged_path,
            "agg_by_model": agg_model_path,
            "agg_by_d0_bins": agg_bins_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "merge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Merge concluído")
    print(f"- {merged_path}")
    print(f"- {agg_model_path}")
    print(f"- {agg_bins_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
