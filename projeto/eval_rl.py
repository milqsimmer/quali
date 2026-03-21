import os, json, csv, time, argparse
import numpy as np
import gym
import argparse
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from two_link_arm_env import TwoLinkArmEnv


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    suffix = f"_" + run_tag if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}")


def get_model_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), f"ppo_model_{seed}.zip")


def evaluate(
    mode: str,
    episodes: int = 100,
    render: bool = False,
    max_steps: int = 200,
    print_episodes: bool = False,
    seed: int = 0,
    run_tag: str = "",
):
    """Avalia um modelo salvo para um determinado modo ('pure' ou 'pirl')."""
    model_path = get_model_path(mode, seed, run_tag)
    print(f"Carregando modelo do caminho: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # env legado (gym 0.21): reset()->obs ; step()->(obs, reward, done, info)
    env = TwoLinkArmEnv(render=render, reward_mode=mode)
    env = TimeLimit(env, max_episode_steps=max_steps)
    model = PPO.load(model_path, env=env)

    # tenta descobrir o dt do ambiente base (TwoLinkArmEnv)
    base_env = env
    if hasattr(base_env, "env"):
        base_env = base_env.env
    dt = getattr(base_env, "dt", 1.0 / 240.0)

    ep_rows = []
    succ = 0
    dists, lens = [], []
    efforts = []  # lista de E_i por episódio
    energies = []  # energia total aproximada por episódio

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        ep_tau_sum = 0.0  # soma_t ||tau_t||_1 no episódio
        ep_energy = 0.0  # soma_t power_t * dt no episódio
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            last_info = info

            tau_sum = float(info.get("tau_sum", np.nan))
            if not np.isnan(tau_sum):
                ep_tau_sum += tau_sum

            power = float(info.get("power", np.nan))
            if not np.isnan(power):
                ep_energy += power * dt

            if render:
                time.sleep(1 / 240.0)

        final_dist = float(last_info.get("distance", np.nan))

        if final_dist < 0.05:
            succ += 1

        # esforço médio em torque: E_i = (1/T_i) * sum_t ||tau_t||_1
        if ep_len > 0:
            E_i = ep_tau_sum / ep_len
        else:
            E_i = np.nan

        dists.append(final_dist)
        lens.append(ep_len)
        efforts.append(E_i)
        energies.append(ep_energy)

        if print_episodes:
            print(
                f"[{mode}] ep {ep+1}/{episodes} - "
                f"dist_final={final_dist:.4f}, ret={ep_ret:.2f}, "
                f"len={ep_len}, E_i={E_i:.4f}"
            )

        ep_rows.append(
            {
                "mode": mode,
                "seed": seed,
                "episode": ep + 1,
                "success": int(final_dist < 0.05),
                "final_distance": final_dist,
                "return": ep_ret,
                "length": ep_len,
                "mean_tau_sum": E_i,  # esforço médio em torque
                "tau_sum_total": ep_tau_sum,  # soma de ||tau_t||_1 no episódio
                "energy": ep_energy,
            }
        )

    env.close()

    summary = {
        "mode": mode,
        "seed": seed,
        "episodes": episodes,
        "success_rate": succ / episodes,
        "mean_final_dist": float(np.nanmean(dists)),
        "std_final_dist": float(np.nanstd(dists)),
        "mean_ep_len": float(np.mean(lens)),
        "std_ep_len": float(np.std(lens)),
        "mean_tau_effort": float(np.nanmean(efforts)),
        "std_tau_effort": float(np.nanstd(efforts)),
        "median_tau_effort": float(np.nanmedian(efforts)),
        "mean_energy": float(np.nanmean(energies)),
        "std_energy": float(np.nanstd(energies)),
    }
    return summary, ep_rows


def save_csv(csv_path: str, rows: list, summaries: list):
    dir_name = os.path.dirname(csv_path)
    if dir_name:  # só cria diretório se tiver pasta no caminho
        os.makedirs(dir_name, exist_ok=True)
    # salva por-episódio
    fieldnames = [
        "mode",
        "seed",
        "episode",
        "success",
        "final_distance",
        "return",
        "length",
        "mean_tau_sum",
        "tau_sum_total",
        "energy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # salva um segundo arquivo com os resumos
    base, ext = os.path.splitext(csv_path)
    sum_path = base + "_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    return csv_path, sum_path


def print_summary_table(summaries: list):
    # impressão amigável lado a lado
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    headers = [
        "mode",
        "seed",
        "episodes",
        "success_rate",
        "mean_final_dist",
        "mean_ep_len",
        "mean_tau_effort",
        "mean_energy",
    ]
    widths = [max(len(h), 14) for h in headers]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for s in summaries:
        row = [
            s["mode"],
            s["seed"],
            s["episodes"],
            fmt(s["success_rate"]),
            fmt(s["mean_final_dist"]),
            fmt(s["mean_ep_len"]),
            fmt(s.get("mean_tau_effort", float("nan"))),
            fmt(s.get("mean_energy", float("nan"))),
        ]
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["pure", "pirl", "both"],
        default="both",
        help="Qual modelo avaliar.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed para reproducibilidade.")
    ap.add_argument("--episodes", type=int, default=100, help="Episódios por modo.")
    ap.add_argument(
        "--render", action="store_true", help="Renderiza durante avaliação."
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Limite de passos por episódio (TimeLimit).",
    )
    ap.add_argument(
        "--out", default="results/eval_results", help="Caminho do CSV por-episódio."
    )
    ap.add_argument(
        "--print-episodes",
        action="store_true",
        help="Imprime métricas por episódio (só para o modo selecionado).",
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Sufixo opcional usado no diretório de treino (seed_X_<tag>).",
    )

    args = ap.parse_args()

    modes = ["pure", "pirl"] if args.mode == "both" else [args.mode]

    all_rows = []
    summaries = []
    for m in modes:
        summary, rows = evaluate(
            m,
            episodes=args.episodes,
            render=args.render,
            max_steps=args.max_steps,
            print_episodes=args.print_episodes,
            seed=args.seed,
            run_tag=args.run_tag,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    print_summary_table(summaries)
    csv_path, sum_path = save_csv(f"{args.out}_{args.seed}.csv", all_rows, summaries)
    print(f"Arquivos salvos:- CSV por-episódio: {csv_path}- Resumo JSON: {sum_path}")
