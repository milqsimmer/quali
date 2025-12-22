# eval_rl.py (A3) — avalia com eval_seed_base + episodes, salva CSV + summary JSON
import os
import json
import csv
import time
import argparse

import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from two_link_arm_env import TwoLinkArmEnv


def _make_env(
    mode: str,
    render: bool,
    max_steps: int,
    base_seed: int,
    margin: float,
    min_tip_dist: float,
    phi_min: float,
    phi_max: float,
):
    base_env = TwoLinkArmEnv(
        render=render,
        reward_mode=mode,
        seed=base_seed,
        margin=margin,
        min_tip_dist=min_tip_dist,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    env = TimeLimit(base_env, max_episode_steps=max_steps)
    env = Monitor(env)
    return env


def evaluate_model(
    mode: str,
    train_seed: int,
    eval_seed_base: int,
    episodes: int,
    render: bool,
    max_steps: int,
    margin: float,
    min_tip_dist: float,
    phi_min: float,
    phi_max: float,
    success_tol: float,
    print_episodes: bool,
):
    model_path = f"runs_{mode}/ppo_model_seed{train_seed}.zip"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    env = _make_env(
        mode=mode,
        render=render,
        max_steps=max_steps,
        base_seed=eval_seed_base,
        margin=margin,
        min_tip_dist=min_tip_dist,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    model = PPO.load(model_path, env=env)

    rows = []
    succ = 0

    final_dists, returns, lengths = [], [], []
    mean_tau_efforts, d0s = [], []

    for ep in range(episodes):
        ep_seed = eval_seed_base + ep
        obs = env.reset(seed=ep_seed)

        d0 = float(getattr(env.unwrapped, "d0", np.nan))
        d0s.append(d0)

        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_r_dist = 0.0
        ep_r_act = 0.0
        ep_r_tau = 0.0
        tau_sum = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_r_dist += float(info.get("r_dist", 0.0))
            ep_r_act += float(info.get("r_act", 0.0))
            ep_r_tau += float(info.get("r_tau", 0.0))
            ep_ret += float(reward)
            ep_len += 1
            last_info = info

            tau = info.get("tau_l1", None)
            if tau is not None:
                tau_sum += float(tau)

            if render:
                time.sleep(1 / 240.0)

        final_dist = float(last_info.get("distance", np.nan))
        success = int(final_dist <= success_tol)
        succ += success

        mean_tau = (tau_sum / ep_len) if ep_len > 0 else np.nan

        final_dists.append(final_dist)
        returns.append(ep_ret)
        lengths.append(ep_len)
        mean_tau_efforts.append(mean_tau)

        if print_episodes:
            print(
                f"[{mode}] train_seed={train_seed} ep {ep+1}/{episodes} "
                f"eval_seed={ep_seed} | d0={d0:.4f} final_dist={final_dist:.4f} "
                f"ret={ep_ret:.2f} len={ep_len} mean_tau={mean_tau:.4f} success={success}"
            )

        rows.append(
            {
                "mode": mode,
                "train_seed": train_seed,
                "episode": ep + 1,
                "eval_seed": ep_seed,
                "success": success,
                "d0": d0,
                "final_distance": final_dist,
                "return": ep_ret,
                "length": ep_len,
                "mean_tau_l1": mean_tau,
                "tau_l1_sum": tau_sum,
                "margin": margin,
                "min_tip_dist": min_tip_dist,
                "phi_min": phi_min,
                "phi_max": phi_max,
                "max_steps": max_steps,
                "success_tol": success_tol,
                "sum_r_dist": ep_r_dist,
                "sum_r_act": ep_r_act,
                "sum_r_tau": ep_r_tau,
                "mean_r_dist": ep_r_dist / ep_len if ep_len > 0 else np.nan,
                "mean_r_act": ep_r_act / ep_len if ep_len > 0 else np.nan,
                "mean_r_tau": ep_r_tau / ep_len if ep_len > 0 else np.nan,
            }
        )

    env.close()

    summary = {
        "mode": mode,
        "train_seed": train_seed,
        "episodes": episodes,
        "eval_seed_base": eval_seed_base,
        "eval_seeds_range": [eval_seed_base, eval_seed_base + episodes - 1],
        "margin": margin,
        "min_tip_dist": min_tip_dist,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "max_steps": max_steps,
        "success_tol": success_tol,
        "success_rate": succ / max(1, episodes),
        "mean_d0": float(np.nanmean(d0s)),
        "std_d0": float(np.nanstd(d0s)),
        "mean_final_dist": float(np.nanmean(final_dists)),
        "std_final_dist": float(np.nanstd(final_dists)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_len": float(np.mean(lengths)),
        "std_ep_len": float(np.std(lengths)),
        "mean_tau_effort": float(np.nanmean(mean_tau_efforts)),
        "std_tau_effort": float(np.nanstd(mean_tau_efforts)),
    }
    return summary, rows


def save_outputs(out_csv: str, all_rows: list, summaries: list):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    base, _ = os.path.splitext(out_csv)
    out_json = base + "_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return out_csv, out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "pirl", "both"], default="both")
    ap.add_argument(
        "--train-seed", type=int, default=0, help="seed do treino (nome do zip)"
    )

    ap.add_argument("--eval-seed-base", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=200)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default="results/eval_results.csv")

    # A2 params (devem bater com o treino p/ comparação justa)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-tip-dist", type=float, default=0.07)
    ap.add_argument("--phi-min", type=float, default=-np.pi / 2)
    ap.add_argument("--phi-max", type=float, default=np.pi / 2)

    ap.add_argument("--success-tol", type=float, default=0.05)
    ap.add_argument("--print-episodes", action="store_true")

    args = ap.parse_args()

    modes = ["pure", "pirl"] if args.mode == "both" else [args.mode]

    all_rows = []
    summaries = []

    for m in modes:
        summary, rows = evaluate_model(
            mode=m,
            train_seed=args.train_seed,
            eval_seed_base=args.eval_seed_base,
            episodes=args.episodes,
            render=args.render,
            max_steps=args.max_steps,
            margin=args.margin,
            min_tip_dist=args.min_tip_dist,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            success_tol=args.success_tol,
            print_episodes=args.print_episodes,
        )
        summaries.append(summary)
        all_rows.extend(rows)

        print(
            f"[{summary['mode']}] train_seed={summary['train_seed']} "
            f"episodes={summary['episodes']} eval_seed_base={summary['eval_seed_base']} "
            f"success_rate={summary['success_rate']:.3f} mean_final_dist={summary['mean_final_dist']:.4f} "
            f"mean_tau_effort={summary['mean_tau_effort']:.4f}"
        )

    out_csv, out_json = save_outputs(args.out, all_rows, summaries)
    print(f"\nArquivos salvos:\n- CSV: {out_csv}\n- Summary: {out_json}")


if __name__ == "__main__":
    main()
