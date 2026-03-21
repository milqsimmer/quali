import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO

from two_link_arm_env import TwoLinkArmEnv


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    suffix = f"_" + run_tag if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}")


def get_model_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), f"ppo_model_{seed}.zip")


def run_one(
    mode: str,
    policy: str,
    episodes: int,
    max_steps: int,
    seed: int,
    run_tag: str,
):
    """Roda varios episodios e coleta escalas tipicas de dist, acao, torque e energia."""

    # env legado (gym 0.21): reset()->obs ; step()->(obs, reward, done, info)
    env = TwoLinkArmEnv(render=False, reward_mode=mode)
    env = TimeLimit(env, max_episode_steps=max_steps)

    # descobre dt do ambiente base (TwoLinkArmEnv)
    base_env = env
    if hasattr(base_env, "env"):
        base_env = base_env.env
    dt = getattr(base_env, "dt", 1.0 / 240.0)

    model = None
    if policy == "trained":
        model_path = get_model_path(mode, seed, run_tag)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
        model = PPO.load(model_path, env=env)

    dists = []
    act_norms = []
    tau_sums = []
    powers = []
    energies = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_energy = 0.0

        while not done:
            if policy == "trained":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)

            # distancia
            dist = float(info.get("distance", np.nan))
            dists.append(dist)

            # norma da acao
            act_norm = float(np.linalg.norm(action))
            act_norms.append(act_norm)

            # soma de torques
            tau_sum = float(info.get("tau_sum", np.nan))
            tau_sums.append(tau_sum)

            # potencia instantanea
            power = float(info.get("power", np.nan))
            powers.append(power)

            if not np.isnan(power):
                ep_energy += power * dt

        energies.append(ep_energy)

    env.close()

    def stats(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)), float(np.nanstd(x))

    results = {
        "mode": mode,
        "policy": policy,
        "dt": float(dt),
        "mean_dist": stats(dists)[0],
        "std_dist": stats(dists)[1],
        "mean_act_norm": stats(act_norms)[0],
        "std_act_norm": stats(act_norms)[1],
        "mean_tau_sum": stats(tau_sums)[0],
        "std_tau_sum": stats(tau_sums)[1],
        "mean_power": stats(powers)[0],
        "std_power": stats(powers)[1],
        "mean_energy": stats(energies)[0],
        "std_energy": stats(energies)[1],
    }
    return results


def make_barplot_two_modes(
    results_pure: dict,
    results_pirl: dict,
    out_prefix: str,
):
    """Gera graficos simples comparando pure vs pirl."""

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    modes = ["pure", "pirl"]
    labels = ["RL puro", "PIRL"]
    xs = np.arange(len(modes))

    # 1) Distancia media
    dist_means = [results_pure["mean_dist"], results_pirl["mean_dist"]]
    plt.figure()
    plt.bar(xs, dist_means, tick_label=labels)
    plt.ylabel("Distancia media por passo (m)")
    plt.title("Distancia media - pure vs pirl")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mean_dist.png", dpi=300)
    plt.close()

    # 2) Esforco medio em torque (tau_sum medio por passo)
    tau_means = [results_pure["mean_tau_sum"], results_pirl["mean_tau_sum"]]
    plt.figure()
    plt.bar(xs, tau_means, tick_label=labels)
    plt.ylabel("Soma media de torques |tau1|+|tau2|")
    plt.title("Esforco medio em torque - pure vs pirl")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mean_tau_sum.png", dpi=300)
    plt.close()

    # 3) Energia media por episodio
    energy_means = [results_pure["mean_energy"], results_pirl["mean_energy"]]
    plt.figure()
    plt.bar(xs, energy_means, tick_label=labels)
    plt.ylabel("Energia media por episodio (unid. arbitrarias)")
    plt.title("Energia media - pure vs pirl")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mean_energy.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analisa escalas tipicas de distancia, acao, torque, potencia e "
            "energia para o ambiente TwoLinkArm, em modos pure/pirl."
        )
    )
    ap.add_argument(
        "--mode",
        choices=["pure", "pirl", "both"],
        default="both",
        help="Qual modo analisar (pure, pirl ou both).",
    )
    ap.add_argument(
        "--policy",
        choices=["random", "trained"],
        default="trained",
        help="Politica usada: modelo PPO treinado ou acoes aleatorias.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed / indice do modelo (para policy=trained).",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Numero de episodios por modo.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Limite de passos por episodio (TimeLimit).",
    )
    ap.add_argument(
        "--out-prefix",
        default="results/analyze_scales",
        help="Prefixo para salvar figuras (ex.: results/analyze_scales).",
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Sufixo opcional usado no diretorio de treino (seed_X_<tag>).",
    )

    args = ap.parse_args()

    if args.mode == "both":
        modes = ["pure", "pirl"]
    else:
        modes = [args.mode]

    all_results = {}

    for m in modes:
        print(f"\n=== Rodando analise para modo={m}, policy={args.policy} ===")
        res = run_one(
            mode=m,
            policy=args.policy,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            run_tag=args.run_tag,
        )
        all_results[m] = res

        print(
            f"dt={res['dt']:.6f} | "
            f"mean_dist={res['mean_dist']:.4f} +- {res['std_dist']:.4f} | "
            f"mean_act_norm={res['mean_act_norm']:.4f} +- {res['std_act_norm']:.4f}"
        )
        print(
            f"mean_tau_sum={res['mean_tau_sum']:.4f} +- {res['std_tau_sum']:.4f} | "
            f"mean_power={res['mean_power']:.4f} +- {res['std_power']:.4f}"
        )
        print(
            f"mean_energy={res['mean_energy']:.4f} +- {res['std_energy']:.4f}"
        )

    # graficos de comparacao pure vs pirl (se tivermos ambos)
    if "pure" in all_results and "pirl" in all_results:
        make_barplot_two_modes(
            all_results["pure"],
            all_results["pirl"],
            out_prefix=args.out_prefix,
        )
        print(
            f"\nFiguras salvas com prefixo '{args.out_prefix}_*.png' "
            "(distancia media, torque medio, energia media)."
        )


if __name__ == "__main__":
    main()
