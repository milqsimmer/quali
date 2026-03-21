import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    print("\n=== Executando:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Roda treinamentos (pure e pirl) e avaliacoes para um intervalo de seeds "
            "usando train_rl.py e eval_rl.py."
        )
    )
    parser.add_argument("--first-seed", type=int, default=0, help="Seed inicial (inclusive).")
    parser.add_argument("--last-seed", type=int, default=5, help="Seed final (inclusive).")
    parser.add_argument(
        "--steps",
        type=int,
        default=300_000,
        help="Numero de passos de treino por seed.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Numero de episodios por seed na avaliacao.",
    )
    parser.add_argument(
        "--lambda-a",
        type=float,
        default=0.001,
        dest="lambda_a",
        help="Peso lambda_a da penalidade de acao.",
    )
    parser.add_argument(
        "--alpha-tau",
        type=float,
        default=0.0005,
        help="Peso alpha_tau da penalidade de torque no modo pirl.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Comando do interpretador Python (padrao: o atual).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Pula a etapa de treino (util se os modelos ja estiverem treinados).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Pula a etapa de avaliacao.",
    )

    args = parser.parse_args()

    if args.first_seed > args.last_seed:
        raise SystemExit("first-seed deve ser <= last-seed")

    py = args.python_exe

    for seed in range(args.first_seed, args.last_seed + 1):
        print("\n" + "=" * 80)
        print(f"Seed {seed}")
        print("=" * 80)

        # 1) Treinos
        if not args.skip_train:
            # modo pure
            cmd_pure = [
                py,
                "train_rl.py",
                "--mode",
                "pure",
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--lambda-a",
                str(args.lambda_a),
            ]
            run_cmd(cmd_pure)

            # modo pirl
            cmd_pirl = [
                py,
                "train_rl.py",
                "--mode",
                "pirl",
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--lambda-a",
                str(args.lambda_a),
                "--alpha-tau",
                str(args.alpha_tau),
            ]
            run_cmd(cmd_pirl)

        # 2) Avaliacao
        if not args.skip_eval:
            # avalia pure e pirl juntos para esta seed
            out_prefix = f"results/eval_official_seed{seed}"
            cmd_eval = [
                py,
                "eval_rl.py",
                "--mode",
                "both",
                "--seed",
                str(seed),
                "--episodes",
                str(args.episodes),
                "--out",
                out_prefix,
            ]
            run_cmd(cmd_eval)


if __name__ == "__main__":
    main()
