# make_latex_tables_d0bins.py
# Gera tabelas LaTeX a partir de aggregated_d0bin_across_seeds.csv
#
# Uso:
#   python make_latex_tables_d0bins.py --in-csv results/aggregated_d0bin_across_seeds.csv --out-tex results/table_d0bins.tex
#   python make_latex_tables_d0bins.py --in-csv results/aggregated_d0bin_across_seeds.csv --out-dir results/tables_d0 --split
#
# Requer: pandas, numpy

from __future__ import annotations

import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd


def fmt_pm(df: pd.DataFrame, mean_col: str, ci_col: str, digits: int) -> pd.Series:
    if mean_col not in df.columns or ci_col not in df.columns:
        return pd.Series([""] * len(df))
    m = pd.to_numeric(df[mean_col], errors="coerce")
    c = pd.to_numeric(df[ci_col], errors="coerce")

    def _one(x, y):
        if not np.isfinite(x):
            return ""
        if not np.isfinite(y):
            return f"{x:.{digits}f}"
        return f"{x:.{digits}f} ± {y:.{digits}f}"

    return pd.Series([_one(x, y) for x, y in zip(m, c)])


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\. ]+", "_", str(s))
    s = s.replace(" ", "_")
    return s


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Colunas formatadas (média ± IC95 sobre seeds)
    out["success"] = fmt_pm(out, "success_rate__mean", "success_rate__ci95", 3)
    out["final_dist"] = fmt_pm(
        out, "mean_final_distance__mean", "mean_final_distance__ci95", 3
    )
    out["tau_l1"] = fmt_pm(out, "mean_tau_l1__mean", "mean_tau_l1__ci95", 3)
    out["filt_rate"] = fmt_pm(
        out,
        "mean_filter_intervention_rate__mean",
        "mean_filter_intervention_rate__ci95",
        3,
    )

    # Ordem de colunas (enxuta)
    cols = [
        "d0_bin",
        "exp_id",
        "n_train_seeds",
        "episodes_in_bin__mean_per_seed",
        "success",
        "final_dist",
        "tau_l1",
        "filt_rate",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    # Ordena: por bin e por performance (se dá)
    if "success" in out.columns:
        # não dá pra ordenar por success textual; usa success_rate__mean se existir
        if "success_rate__mean" in df.columns:
            out = out.join(df["success_rate__mean"])
            out = out.sort_values(
                ["d0_bin", "success_rate__mean", "exp_id"],
                ascending=[True, False, True],
            )
            out = out.drop(columns=["success_rate__mean"])
        else:
            out = out.sort_values(["d0_bin", "exp_id"])
    else:
        out = out.sort_values(["d0_bin", "exp_id"])

    return out


def write_tex(df: pd.DataFrame, out_path: str, caption: str, label: str):
    tex = df.to_latex(index=False, escape=True)
    # embrulho mínimo com caption/label (opcionalmente você ajusta no LaTeX depois)
    wrapped = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tex}\n"
        "\\end{table}\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv", required=True, help="CSV aggregated_d0bin_across_seeds.csv"
    )
    ap.add_argument(
        "--out-tex", default="", help="Arquivo .tex único (se não usar --split)"
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Diretório para salvar um .tex por bin (com --split)",
    )
    ap.add_argument("--split", action="store_true", help="Gera um .tex por d0_bin")
    ap.add_argument(
        "--caption",
        default="Resultados por faixa de dificuldade ($d_0$). Valores: média ± IC95 sobre seeds.",
    )
    ap.add_argument("--label", default="tab:results-d0bins")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "d0_bin" not in df.columns:
        raise SystemExit(
            "CSV não tem coluna d0_bin. Use o aggregate_results.py com binning de d0."
        )

    table = make_table(df)

    if args.split:
        if not args.out_dir:
            raise SystemExit("Use --out-dir quando usar --split.")
        os.makedirs(args.out_dir, exist_ok=True)

        for d0_bin, g in table.groupby("d0_bin", dropna=False):
            name = sanitize_filename(str(d0_bin))
            out_path = os.path.join(args.out_dir, f"table_d0bin_{name}.tex")
            caption = f"Resultados no bin de dificuldade $d_0$ = {d0_bin}. (média ± IC95 sobre seeds)"
            label = f"{args.label}-{name}"
            write_tex(g, out_path, caption=caption, label=label)

        print("[OK] Tabelas por bin geradas em:", args.out_dir)
        return

    # single file
    if not args.out_tex:
        raise SystemExit(
            "Use --out-tex (arquivo único) ou --split --out-dir (um por bin)."
        )

    write_tex(table, args.out_tex, caption=args.caption, label=args.label)
    print("[OK] Tabela única gerada em:", args.out_tex)


if __name__ == "__main__":
    main()
