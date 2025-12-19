# train_many_seeds.py
import argparse
import os
import subprocess
import sys


def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "pirl"], default="pure")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--render", action="store_true")

    # parâmetros do A2 (se você adicionou no train_rl)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min_tip_dist", type=float, default=0.07)

    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    py = sys.executable

    for seed in seeds:
        cmd = [
            py,
            "train_rl.py",
            "--mode",
            args.mode,
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
            "--margin",
            str(args.margin),
            "--min_tip_dist",
            str(args.min_tip_dist),
        ]
        if args.render:
            cmd.append("--render")

        run(cmd)

    print("\n[OK] Treinos finalizados para seeds:", seeds)
