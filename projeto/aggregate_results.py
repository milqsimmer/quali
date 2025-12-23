# aggregate_results.py
# Agrega resultados do eval_rl.py:
# 1) Geral: por (exp_id, train_seed) e depois por exp_id (média/DP/IC95 sobre seeds)
# 2) Por bins de dificuldade (d0): por (exp_id, train_seed, d0_bin) e depois por (exp_id, d0_bin)
#
# Uso:
#   python aggregate_results.py --results-dir results --out-dir results
#   python aggregate_results.py --results-dir results --out-dir results --d0-quantiles "0,0.33,0.66,1"
#   python aggregate_results.py --results-dir results --out-dir results --d0-bins "0,0.35,0.7,1.2"
#   python aggregate_results.py --results-dir results --out-dir results --latex results/table_across_seeds.tex
#
# Requer: pandas, numpy

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EVAL_CSV_RE = re.compile(
    r"eval__(?P<exp_id>.+)__trainseed(?P<train_seed>\d+)__evalbase(?P<eval_base>\d+)\.csv$"
)


@dataclass
class FileMeta:
    path: str
    exp_id: str
    train_seed: int
    eval_base: int


def find_eval_csvs(results_dir: str) -> List[FileMeta]:
    paths = glob.glob(os.path.join(results_dir, "eval__*__trainseed*__evalbase*.csv"))
    metas: List[FileMeta] = []
    for p in paths:
        base = os.path.basename(p)
        m = EVAL_CSV_RE.match(base)
        if not m:
            continue
        metas.append(
            FileMeta(
                path=p,
                exp_id=m.group("exp_id"),
                train_seed=int(m.group("train_seed")),
                eval_base=int(m.group("eval_base")),
            )
        )
    metas.sort(key=lambda x: (x.exp_id, x.train_seed, x.eval_base, x.path))
    return metas


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_mean(x: pd.Series) -> float:
    x = _to_num(x)
    return float(np.nanmean(x.to_numpy(dtype=float)))


def safe_std(x: pd.Series) -> float:
    x = _to_num(x)
    n = int(x.notna().sum())
    return float(np.nanstd(x.to_numpy(dtype=float), ddof=1)) if n > 1 else float("nan")


def ci95(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return float("nan")
    return float(1.96 * std / np.sqrt(n))


def summarize_one_file(df: pd.DataFrame, meta: FileMeta) -> Dict:
    out: Dict = {
        "exp_id": meta.exp_id,
        "train_seed": meta.train_seed,
        "eval_seed_base": meta.eval_base,
        "episodes": int(len(df)),
        "source_file": os.path.basename(meta.path),
    }

    # Primárias
    if "success" in df.columns:
        out["success_rate"] = safe_mean(df["success"])
    if "final_distance" in df.columns:
        out["mean_final_distance"] = safe_mean(df["final_distance"])
        out["std_final_distance_ep"] = safe_std(df["final_distance"])
    if "return" in df.columns:
        out["mean_return"] = safe_mean(df["return"])
        out["std_return_ep"] = safe_std(df["return"])
    if "length" in df.columns:
        out["mean_ep_len"] = safe_mean(df["length"])
        out["std_ep_len_ep"] = safe_std(df["length"])

    # dificuldade
    if "d0" in df.columns:
        out["mean_d0"] = safe_mean(df["d0"])
        out["std_d0_ep"] = safe_std(df["d0"])

    # esforço / PI / safety
    if "mean_tau_l1" in df.columns:
        out["mean_tau_l1"] = safe_mean(df["mean_tau_l1"])
        out["std_tau_l1_ep"] = safe_std(df["mean_tau_l1"])
    if "mean_pi_value" in df.columns:
        out["mean_pi_value"] = safe_mean(df["mean_pi_value"])
        out["std_pi_value_ep"] = safe_std(df["mean_pi_value"])

    if "filter_intervention_rate" in df.columns:
        out["mean_filter_intervention_rate"] = safe_mean(df["filter_intervention_rate"])
        out["std_filter_intervention_rate_ep"] = safe_std(
            df["filter_intervention_rate"]
        )
    if "mean_filter_delta_tau_norm" in df.columns:
        out["mean_filter_delta_tau_norm"] = safe_mean(df["mean_filter_delta_tau_norm"])
        out["std_filter_delta_tau_norm_ep"] = safe_std(df["mean_filter_delta_tau_norm"])

    # decomposição (se existir)
    for k in [
        "mean_r_dist",
        "mean_r_act",
        "mean_r_pi",
        "sum_r_dist",
        "sum_r_act",
        "sum_r_pi",
    ]:
        if k in df.columns:
            out[k] = safe_mean(df[k])

    # eixos (pega 1º não nulo; deve ser constante)
    for k in ["control", "use_pi_reward", "pi_metric", "alpha_pi", "safety_filter"]:
        if k in df.columns:
            nonnull = df[k].dropna()
            out[k] = nonnull.iloc[0] if len(nonnull) else ""

    return out


def aggregate_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["exp_id"]
    for k in ["control", "use_pi_reward", "pi_metric", "alpha_pi", "safety_filter"]:
        if k in per_seed.columns:
            group_cols.append(k)

    metrics = [
        "success_rate",
        "mean_final_distance",
        "mean_return",
        "mean_ep_len",
        "mean_tau_l1",
        "mean_pi_value",
        "mean_filter_intervention_rate",
        "mean_filter_delta_tau_norm",
        "mean_d0",
        "mean_r_dist",
        "mean_r_act",
        "mean_r_pi",
        "sum_r_dist",
        "sum_r_act",
        "sum_r_pi",
    ]
    metrics = [m for m in metrics if m in per_seed.columns]

    rows = []
    for key, g in per_seed.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        n = int(g["train_seed"].nunique()) if "train_seed" in g.columns else int(len(g))
        base["n_train_seeds"] = n

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            mu = float(np.nanmean(vals))
            sd = (
                float(np.nanstd(vals, ddof=1))
                if np.isfinite(vals).sum() > 1
                else float("nan")
            )
            base[f"{m}__mean"] = mu
            base[f"{m}__std"] = sd
            base[f"{m}__ci95"] = ci95(sd, n)

        rows.append(base)

    out = pd.DataFrame(rows)
    if "success_rate__mean" in out.columns:
        out = out.sort_values(["success_rate__mean", "exp_id"], ascending=[False, True])
    else:
        out = out.sort_values(["exp_id"])
    return out


# ------------------- d0 binning (por episódio) -------------------


def load_all_raw_rows(metas: List[FileMeta], require_episodes: int = 0) -> pd.DataFrame:
    frames = []
    for meta in metas:
        df = pd.read_csv(meta.path)
        if require_episodes > 0 and len(df) != require_episodes:
            continue
        df["exp_id"] = meta.exp_id
        df["train_seed"] = meta.train_seed
        df["eval_seed_base"] = meta.eval_base
        df["source_file"] = os.path.basename(meta.path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_d0_bins(
    raw: pd.DataFrame,
    d0_bins: Optional[List[float]],
    d0_quantiles: Optional[List[float]],
) -> Tuple[np.ndarray, List[str]]:
    if "d0" not in raw.columns:
        raise ValueError(
            "Não existe coluna d0 nos CSVs. O binning por dificuldade requer d0 por episódio."
        )

    d0 = pd.to_numeric(raw["d0"], errors="coerce")
    d0 = d0[np.isfinite(d0)]
    if len(d0) == 0:
        raise ValueError("d0 está todo NaN/inf; não dá para criar bins.")

    if d0_bins is not None:
        edges = np.array(d0_bins, dtype=float)
        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                "d0-bins precisa ser estritamente crescente, ex: 0,0.3,0.6,1.0"
            )
    else:
        qs = d0_quantiles if d0_quantiles is not None else [0.0, 0.33, 0.66, 1.0]
        qs = np.array(qs, dtype=float)
        if qs[0] != 0.0 or qs[-1] != 1.0:
            raise ValueError(
                "d0-quantiles deve começar em 0 e terminar em 1. Ex: 0,0.33,0.66,1"
            )
        edges = np.quantile(d0.to_numpy(), qs, method="linear")
        # garantir monotonicidade (caso d0 repita muito)
        edges = np.unique(edges)
        if len(edges) < 3:
            # fallback: bins fixos bem genéricos
            edges = np.array(
                [float(d0.min()), float(d0.mean()), float(d0.max())], dtype=float
            )

    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"[{edges[i]:.3f},{edges[i+1]:.3f})")
    labels[-1] = labels[-1].replace(")", "]")  # último fechado
    return edges, labels


def add_d0_bin(raw: pd.DataFrame, edges: np.ndarray, labels: List[str]) -> pd.DataFrame:
    raw = raw.copy()
    raw["d0"] = pd.to_numeric(raw["d0"], errors="coerce")
    raw["d0_bin"] = pd.cut(
        raw["d0"], bins=edges, labels=labels, include_lowest=True, right=False
    )
    # fecha o último intervalo à direita (para incluir o max)
    # workaround: re-cut with right=True only for last edge? Simples: ajustar d0==max para dentro
    d0_max = float(np.nanmax(raw["d0"].to_numpy(dtype=float)))
    last_label = labels[-1]
    raw.loc[np.isclose(raw["d0"], d0_max, atol=1e-12), "d0_bin"] = last_label
    return raw


def aggregate_d0bin_per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    # métricas por episódio -> agregamos por (exp_id, train_seed, d0_bin)
    metrics = {
        "success_rate": ("success", "mean"),
        "mean_final_distance": ("final_distance", "mean"),
        "mean_return": ("return", "mean"),
        "mean_ep_len": ("length", "mean"),
        "mean_tau_l1": ("mean_tau_l1", "mean"),
        "mean_pi_value": ("mean_pi_value", "mean"),
        "mean_filter_intervention_rate": ("filter_intervention_rate", "mean"),
        "mean_filter_delta_tau_norm": ("mean_filter_delta_tau_norm", "mean"),
        "mean_d0": ("d0", "mean"),
        "episodes_in_bin": (
            ("episode", "count") if "episode" in raw.columns else ("d0", "count")
        ),
    }

    # manter só colunas existentes
    use = {}
    for out_name, (col, agg) in metrics.items():
        if col in raw.columns:
            use[out_name] = (col, agg)

    group_cols = ["exp_id", "train_seed", "eval_seed_base", "d0_bin"]
    axis_cols = [
        c
        for c in ["control", "use_pi_reward", "pi_metric", "alpha_pi", "safety_filter"]
        if c in raw.columns
    ]

    # pegar eixos por grupo (primeiro valor)
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    agg_dict = {v[0]: v[1] for v in use.values()}
    grouped = raw.groupby(group_cols, dropna=False)

    out = grouped.agg(agg_dict).reset_index()

    # renomear colunas agregadas para os nomes de saída
    rename_map = {}
    for out_name, (col, _) in use.items():
        rename_map[col] = out_name
    out = out.rename(columns=rename_map)

    # anexa eixos
    for c in axis_cols:
        tmp = grouped[c].apply(first_nonnull).reset_index().rename(columns={c: c})
        out = out.merge(tmp, on=group_cols, how="left")

    # ordem
    out = out.sort_values(["exp_id", "d0_bin", "train_seed"])
    return out


def aggregate_d0bin_across_seeds(d0_per_seed: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["exp_id", "d0_bin"]
    for k in ["control", "use_pi_reward", "pi_metric", "alpha_pi", "safety_filter"]:
        if k in d0_per_seed.columns:
            group_cols.append(k)

    metrics = [
        c
        for c in d0_per_seed.columns
        if c.startswith("mean_")
        or c.endswith("_rate")
        or c.endswith("_distance")
        or c in ("success_rate",)
    ]
    # garante os principais
    main_metrics = [
        "success_rate",
        "mean_final_distance",
        "mean_tau_l1",
        "mean_filter_intervention_rate",
        "mean_pi_value",
        "mean_return",
        "mean_ep_len",
        "mean_d0",
    ]
    metrics = [m for m in main_metrics if m in d0_per_seed.columns]

    rows = []
    for key, g in d0_per_seed.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        n = int(g["train_seed"].nunique()) if "train_seed" in g.columns else int(len(g))
        base["n_train_seeds"] = n

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            mu = float(np.nanmean(vals))
            sd = (
                float(np.nanstd(vals, ddof=1))
                if np.isfinite(vals).sum() > 1
                else float("nan")
            )
            base[f"{m}__mean"] = mu
            base[f"{m}__std"] = sd
            base[f"{m}__ci95"] = ci95(sd, n)

        # também agregamos quantos episódios caíram no bin (média por seed e total)
        if "episodes_in_bin" in g.columns:
            ep_vals = pd.to_numeric(g["episodes_in_bin"], errors="coerce").to_numpy(
                dtype=float
            )
            base["episodes_in_bin__mean_per_seed"] = float(np.nanmean(ep_vals))
            base["episodes_in_bin__total"] = float(np.nansum(ep_vals))

        rows.append(base)

    out = pd.DataFrame(rows).sort_values(["exp_id", "d0_bin"])
    return out


def to_latex_table(across: pd.DataFrame, out_path: str):
    # tabela enxuta “pronta”:
    # exp_id | success(mean±ci) | final_dist(mean±ci) | tau_l1(mean±ci) | filt_rate(mean±ci)
    cols = ["exp_id"]

    def fmt(mean_col: str, ci_col: str, digits: int = 3) -> pd.Series:
        m = across.get(mean_col)
        c = across.get(ci_col)
        if m is None or c is None:
            return pd.Series([""] * len(across))
        return (
            m.map(lambda x: f"{x:.{digits}f}")
            + " ± "
            + c.map(lambda x: f"{x:.{digits}f}" if np.isfinite(x) else "nan")
        )

    out = across.copy()
    out["success_rate"] = (
        fmt("success_rate__mean", "success_rate__ci95", 3)
        if "success_rate__mean" in out
        else ""
    )
    out["final_dist"] = (
        fmt("mean_final_distance__mean", "mean_final_distance__ci95", 3)
        if "mean_final_distance__mean" in out
        else ""
    )
    out["tau_l1"] = (
        fmt("mean_tau_l1__mean", "mean_tau_l1__ci95", 3)
        if "mean_tau_l1__mean" in out
        else ""
    )
    out["filt_rate"] = (
        fmt(
            "mean_filter_intervention_rate__mean",
            "mean_filter_intervention_rate__ci95",
            3,
        )
        if "mean_filter_intervention_rate__mean" in out
        else ""
    )

    cols = [
        "exp_id",
        "n_train_seeds",
        "success_rate",
        "final_dist",
        "tau_l1",
        "filt_rate",
    ]
    cols = [c for c in cols if c in out.columns]
    latex = out[cols].to_latex(index=False, escape=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="pasta com eval__*.csv")
    ap.add_argument(
        "--out-dir", default="results", help="pasta para salvar aggregated_*.csv"
    )
    ap.add_argument(
        "--require-episodes",
        type=int,
        default=0,
        help="se >0, ignora arquivos com episódios != esse valor",
    )

    # d0 binning: escolha 1 dos dois (bins fixos ou quantis)
    ap.add_argument("--d0-bins", default="", help="edges fixos, ex: '0,0.35,0.7,1.2'")
    ap.add_argument(
        "--d0-quantiles", default="0,0.33,0.66,1", help="quantis, ex: '0,0.33,0.66,1'"
    )

    ap.add_argument("--per-seed-name", default="aggregated_per_seed.csv")
    ap.add_argument("--across-seeds-name", default="aggregated_across_seeds.csv")
    ap.add_argument("--d0bin-per-seed-name", default="aggregated_d0bin_per_seed.csv")
    ap.add_argument(
        "--d0bin-across-seeds-name", default="aggregated_d0bin_across_seeds.csv"
    )

    ap.add_argument(
        "--latex",
        default="",
        help="se definido, salva tabela LaTeX do across-seeds (geral) nesse path",
    )

    args = ap.parse_args()

    metas = find_eval_csvs(args.results_dir)
    if not metas:
        raise SystemExit(
            f"Nenhum arquivo encontrado em {args.results_dir}: eval__*__trainseed*__evalbase*.csv"
        )

    # --------- agregação geral (por seed / por exp) ---------
    summaries = []
    for meta in metas:
        df = pd.read_csv(meta.path)
        if args.require_episodes > 0 and len(df) != args.require_episodes:
            continue
        summaries.append(summarize_one_file(df, meta))

    if not summaries:
        raise SystemExit("Nenhum arquivo passou no filtro (require-episodes).")

    per_seed = pd.DataFrame(summaries)
    across = aggregate_across_seeds(per_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_per_seed = os.path.join(args.out_dir, args.per_seed_name)
    out_across = os.path.join(args.out_dir, args.across_seeds_name)

    per_seed.to_csv(out_per_seed, index=False, encoding="utf-8")
    across.to_csv(out_across, index=False, encoding="utf-8")

    # --------- agregação por d0 bins (por episódio) ---------
    raw = load_all_raw_rows(metas, require_episodes=args.require_episodes)
    if raw.empty:
        raise SystemExit(
            "Não consegui carregar linhas brutas (raw) para binning de d0."
        )

    d0_bins = parse_list_floats(args.d0_bins) if args.d0_bins.strip() else None
    d0_q = parse_list_floats(args.d0_quantiles) if args.d0_quantiles.strip() else None

    edges, labels = make_d0_bins(raw, d0_bins=d0_bins, d0_quantiles=d0_q)
    raw = add_d0_bin(raw, edges, labels)

    d0_per_seed = aggregate_d0bin_per_seed(raw)
    d0_across = aggregate_d0bin_across_seeds(d0_per_seed)

    out_d0_per_seed = os.path.join(args.out_dir, args.d0bin_per_seed_name)
    out_d0_across = os.path.join(args.out_dir, args.d0bin_across_seeds_name)

    d0_per_seed.to_csv(out_d0_per_seed, index=False, encoding="utf-8")
    d0_across.to_csv(out_d0_across, index=False, encoding="utf-8")

    # --------- latex (opcional) ---------
    if args.latex.strip():
        to_latex_table(across, args.latex.strip())

    print("[OK] Agregação concluída.")
    print(f"- Geral per-seed:         {out_per_seed}")
    print(f"- Geral across-seeds:     {out_across}")
    print(f"- d0-bin per-seed:        {out_d0_per_seed}")
    print(f"- d0-bin across-seeds:    {out_d0_across}")
    if args.latex.strip():
        print(f"- LaTeX table:            {args.latex.strip()}")
    print(f"- d0 bins: edges={edges.tolist()} labels={labels}")


if __name__ == "__main__":
    main()
