# eval_rl.py (A3) — avaliação padronizada, com d0 e seeds por episódio
import os, json, csv, time, argparse
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
        seed=base_seed,  # seed do RNG interno (reforçado no reset)
        margin=margin,
        min_tip_dist=min_tip_dist,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    env = TimeLimit(base_env, max_episode_steps=max_steps)
    env = Monitor(env)  # mantém alinhado ao treino
    return env


def evaluate(
    mode: str,
    model_seed: int,
    episodes: int,
    render: bool,
    max_steps: int,
    eval_seed: int,
    margin: float,
    min_tip_dist: float,
    phi_min: float,
    phi_max: float,
    print_episodes: bool = False,
    success_tol: float = 0.05,
):
    """
    Avalia um modelo salvo em runs_{mode}/ppo_model_seed{model_seed}.zip
    usando seeds por episódio (eval_seed + ep) para garantir comparabilidade.
    """
    model_path = f"runs_{mode}/ppo_model_seed{model_seed}.zip"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # env legado (gym 0.21): reset()->obs ; step()->(obs, reward, done, info)
    env = _make_env(
        mode, render, max_steps, eval_seed, margin, min_tip_dist, phi_min, phi_max
    )
    model = PPO.load(model_path, env=env)

    ep_rows = []
    succ = 0
    dists, rets, lens = [], [], []
    efforts = []
    d0s = []

    for ep in range(episodes):
        episode_seed = eval_seed + ep  # mesmos alvos entre modos
        obs = env.reset(seed=episode_seed)

        # d0 vem do env "desembrulhado"
        d0 = float(getattr(env.unwrapped, "d0", np.nan))
        d0s.append(d0)

        done = False
        ep_ret, ep_len = 0.0, 0
        ep_tau_sum = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_ret += float(reward)
            ep_len += 1
            last_info = info

            tau_l1 = float(info.get("tau_l1", np.nan))
            if not np.isnan(tau_l1):
                ep_tau_sum += tau_l1

            if render:
                time.sleep(1 / 240.0)

        final_dist = float(last_info.get("distance", np.nan))
        success = int(final_dist <= success_tol)
        succ += success

        E_i = (ep_tau_sum / ep_len) if ep_len > 0 else np.nan

        dists.append(final_dist)
        rets.append(ep_ret)
        lens.append(ep_len)
        efforts.append(E_i)

        if print_episodes:
            print(
                f"[{mode}] ep {ep+1}/{episodes} seed={episode_seed} | "
                f"d0={d0:.4f} dist_final={final_dist:.4f} "
                f"ret={ep_ret:.2f} len={ep_len} E_i={E_i:.4f} success={success}"
            )

        ep_rows.append(
            {
                "mode": mode,
                "train_seed": model_seed,
                "eval_seed_base": eval_seed,
                "episode_seed": episode_seed,
                "episode": ep + 1,
                "success": success,
                "d0": d0,
                "final_distance": final_dist,
                "return": ep_ret,
                "length": ep_len,
                "mean_tau_l1": E_i,
                "tau_l1_sum": ep_tau_sum,
                "margin": margin,
                "min_tip_dist": min_tip_dist,
                "phi_min": phi_min,
                "phi_max": phi_max,
                "max_steps": max_steps,
                "success_tol": success_tol,
            }
        )

    env.close()

    summary = {
        "mode": mode,
        "train_seed": model_seed,
        "episodes": episodes,
        "eval_seed_base": eval_seed,
        "margin": margin,
        "min_tip_dist": min_tip_dist,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "max_steps": max_steps,
        "success_tol": success_tol,
        "success_rate": succ / episodes,
        "mean_d0": float(np.nanmean(d0s)),
        "std_d0": float(np.nanstd(d0s)),
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
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    base, _ = os.path.splitext(csv_path)
    sum_path = base + "_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return csv_path, sum_path


def print_summary_table(summaries: list):
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    headers = [
        "mode",
        "train_seed",
        "episodes",
        "success_rate",
        "mean_d0",
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
            s["train_seed"],
            s["episodes"],
            fmt(s["success_rate"]),
            fmt(s["mean_d0"]),
            fmt(s["mean_final_dist"]),
            fmt(s["mean_return"]),
            fmt(s["mean_ep_len"]),
            fmt(s.get("mean_tau_effort", float("nan"))),
        ]
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "pirl", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default="results/eval_results.csv")

    # A3: seeds e parâmetros do alvo
    ap.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="seed usada no treino (nome do arquivo do modelo)",
    )
    ap.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="seed base para gerar episode_seed=eval_seed+ep",
    )

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
        summary, rows = evaluate(
            mode=m,
            model_seed=args.train_seed,
            episodes=args.episodes,
            render=args.render,
            max_steps=args.max_steps,
            eval_seed=args.eval_seed,
            margin=args.margin,
            min_tip_dist=args.min_tip_dist,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            print_episodes=args.print_episodes,
            success_tol=args.success_tol,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    print_summary_table(summaries)
    csv_path, sum_path = save_csv(args.out, all_rows, summaries)
    print(
        f"\nArquivos salvos:\n- CSV por-episódio: {csv_path}\n- Resumo JSON: {sum_path}"
    )
