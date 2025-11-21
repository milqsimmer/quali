import os, json, csv, time, argparse
import numpy as np
import gym
import argparse
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from two_link_arm_env import TwoLinkArmEnv


def evaluate(
    mode: str,
    episodes: int = 100,
    render: bool = False,
    max_steps: int = 200,
    print_episodes: bool = False,
    seed: int = 0,
):
    """Avalia um modelo salvo para um determinado modo ('pure' ou 'pirl')."""
    model_path = f"runs_{mode}/ppo_model_{seed}.zip"
    print(f"Carregando modelo do caminho: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # env legado (gym 0.21): reset()->obs ; step()->(obs, reward, done, info)
    env = TwoLinkArmEnv(render=render, reward_mode=mode)
    env = TimeLimit(env, max_episode_steps=max_steps)
    model = PPO.load(model_path, env=env)

    ep_rows = []
    succ = 0
    dists, rets, lens = [], [], []
    efforts = []  # lista de E_i por episódio

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        ep_tau_sum = 0.0  # soma_t ||tau_t||_1 no episódio
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            last_info = info

            # acumula esforço de torque L1 no episódio
            tau_l1 = float(info.get("tau_l1", np.nan))
            if not np.isnan(tau_l1):
                ep_tau_sum += tau_l1

            if render:
                time.sleep(1 / 240.0)

        final_dist = float(last_info.get("distance", np.nan))

        if final_dist < 0.05:
            succ += 1

        # esforço médio em torque do episódio: E_i = (1/T_i) * sum_t ||tau_t||_1
        if ep_len > 0:
            E_i = ep_tau_sum / ep_len
        else:
            E_i = np.nan

        dists.append(final_dist)
        rets.append(ep_ret)
        lens.append(ep_len)
        efforts.append(E_i)

        ep_rows.append(
            {
                "mode": mode,
                "episode": ep + 1,
                "success": int(final_dist < 0.05),
                "final_dist": final_dist,
                "return": ep_ret,
                "length": ep_len,
                "mean_tau_l1": E_i,  # esforço médio em torque
                "tau_l1_sum": ep_tau_sum,  # soma de ||tau_t||_1 no episódio (opcional)
            }
        )

    env.close()

    summary = {
        "mode": mode,
        "episodes": episodes,
        "success_rate": succ / episodes,
        "mean_final_dist": float(np.nanmean(dists)),
        "std_final_dist": float(np.nanstd(dists)),
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "mean_ep_len": float(np.mean(lens)),
        "std_ep_len": float(np.std(lens)),
        "mean_tau_effort": float(np.nanmean(efforts)),
        "std_tau_effort": float(np.nanstd(efforts)),
        "median_tau_effort": float(np.nanmedian(efforts)),
    }
    return summary, ep_rows


def save_csv(csv_path: str, rows: list, summaries: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # salva por-episódio
    fieldnames = ["mode", "episode", "success", "final_distance", "return", "length"]
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
        "episodes",
        "success_rate",
        "mean_final_dist",
        "mean_return",
        "mean_ep_len",
        "mean_tau_effort",
    ]
    widths = [max(len(h), 14) for h in headers]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for s in summaries:
        row = [
            s["mode"],
            s["episodes"],
            fmt(s["success_rate"]),
            fmt(s["mean_final_dist"]),
            fmt(s["mean_return"]),
            fmt(s["mean_ep_len"]),
            fmt(s.get("mean_tau_effort", float("nan"))),
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
        )
        summaries.append(summary)
        all_rows.extend(rows)

    print_summary_table(summaries)
    csv_path, sum_path = save_csv(f"{args.out}_{args.seed}.csv", all_rows, summaries)
    print(
        f"\nArquivos salvos:\n- CSV por-episódio: {csv_path}\n- Resumo JSON: {sum_path}"
    )
