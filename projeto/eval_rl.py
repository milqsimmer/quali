# eval_rl.py (proposta completa + A3)
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


def load_manifest(exp_id: str) -> dict:
    path = os.path.join("runs_torque", exp_id, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"manifest.json não encontrado para exp_id={exp_id}: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env_from_manifest(manifest: dict, render: bool, base_seed: int):
    e = manifest["env"]
    env = TwoLinkArmEnv(
        render=render,
        seed=base_seed,
        margin=e["margin"],
        min_tip_dist=e["min_tip_dist"],
        phi_min=e["phi_min"],
        phi_max=e["phi_max"],
        success_tol=e["success_tol"],
        tau_max=e["tau_max"],
        lam_a=e["lam_a"],
        use_pi_reward=e["use_pi_reward"],
        pi_metric=e["pi_metric"],
        alpha_pi=e["alpha_pi"],
        safety_filter=e["safety_filter"],
        dtau_max=e["dtau_max"],
        q_margin=e["q_margin"],
        control=e["control"],
        kp=e["kp"],
        kd=e["kd"],
        elbow=e["elbow"],
    )
    env = TimeLimit(env, max_episode_steps=e["max_steps"])
    env = Monitor(env)
    return env


def evaluate(
    exp_id: str,
    train_seed: int,
    eval_seed_base: int,
    episodes: int,
    render: bool,
    print_episodes: bool,
):
    manifest = load_manifest(exp_id)

    model_path = os.path.join("runs_torque", exp_id, f"ppo_model_seed{train_seed}.zip")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    env = make_env_from_manifest(manifest, render=render, base_seed=eval_seed_base)
    model = PPO.load(model_path, env=env)

    rows = []
    succ = 0

    final_dists, returns, lengths = [], [], []
    d0s = []
    tau_efforts = []
    pi_vals = []
    filter_interventions = []
    filter_delta_norms = []
    tau_raw_norms = []
    tau_cmd_norms = []
    tau_nom_norms = []
    tau_res_norms = []

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
        ep_r_pi = 0.0

        tau_sum = 0.0
        pi_sum = 0.0
        filt_count = 0
        filt_delta_sum = 0.0

        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_ret += float(reward)
            ep_len += 1
            last_info = info

            ep_r_dist += float(info.get("r_dist", 0.0))
            ep_r_act += float(info.get("r_act", 0.0))
            ep_r_pi += float(info.get("r_pi", 0.0))

            tau_app1 = float(info.get("tau_app1", 0.0))
            tau_app2 = float(info.get("tau_app2", 0.0))
            tau_sum += abs(tau_app1) + abs(tau_app2)

            pi_sum += float(info.get("pi_value", 0.0))

            filt = int(info.get("filter_intervened", 0))
            filt_count += filt
            filt_delta_sum += float(info.get("filter_delta_tau_norm", 0.0))

            tau_raw = np.array(
                [info.get("tau_raw1", 0.0), info.get("tau_raw2", 0.0)], dtype=float
            )
            tau_cmd = np.array(
                [info.get("tau_cmd1", 0.0), info.get("tau_cmd2", 0.0)], dtype=float
            )

            tau_raw_norms.append(float(np.linalg.norm(tau_raw)))
            tau_cmd_norms.append(float(np.linalg.norm(tau_cmd)))

            tau_nom = np.array(
                [info.get("tau_nom1", np.nan), info.get("tau_nom2", np.nan)],
                dtype=float,
            )
            tau_res = np.array(
                [info.get("tau_res1", np.nan), info.get("tau_res2", np.nan)],
                dtype=float,
            )
            if np.all(np.isfinite(tau_nom)):
                tau_nom_norms.append(float(np.linalg.norm(tau_nom)))
            if np.all(np.isfinite(tau_res)):
                tau_res_norms.append(float(np.linalg.norm(tau_res)))

            if render:
                time.sleep(1 / 240.0)

        final_dist = float(last_info.get("distance", np.nan))
        success = int(final_dist <= float(last_info.get("success_tol", 0.05)))
        succ += success

        mean_tau = (tau_sum / ep_len) if ep_len > 0 else np.nan
        mean_pi = (pi_sum / ep_len) if ep_len > 0 else np.nan
        mean_filt_delta = (filt_delta_sum / max(1, ep_len)) if ep_len > 0 else np.nan

        final_dists.append(final_dist)
        returns.append(ep_ret)
        lengths.append(ep_len)
        tau_efforts.append(mean_tau)
        pi_vals.append(mean_pi)
        filter_interventions.append(filt_count / max(1, ep_len))
        filter_delta_norms.append(mean_filt_delta)

        if print_episodes:
            print(
                f"[{exp_id}] seed={train_seed} ep {ep+1}/{episodes} ep_seed={ep_seed} "
                f"d0={d0:.3f} final={final_dist:.3f} ret={ep_ret:.2f} len={ep_len} "
                f"tau={mean_tau:.3f} pi={mean_pi:.3f} filt%={filter_interventions[-1]:.3f} succ={success}"
            )

        rows.append(
            {
                "exp_id": exp_id,
                "train_seed": train_seed,
                "episode": ep + 1,
                "episode_seed": ep_seed,
                "success": success,
                "d0": d0,
                "final_distance": final_dist,
                "return": ep_ret,
                "length": ep_len,
                "mean_tau_l1": mean_tau,
                "mean_pi_value": mean_pi,
                "filter_intervention_rate": filter_interventions[-1],
                "mean_filter_delta_tau_norm": mean_filt_delta,
                "sum_r_dist": ep_r_dist,
                "sum_r_act": ep_r_act,
                "sum_r_pi": ep_r_pi,
                "mean_r_dist": ep_r_dist / ep_len if ep_len > 0 else np.nan,
                "mean_r_act": ep_r_act / ep_len if ep_len > 0 else np.nan,
                "mean_r_pi": ep_r_pi / ep_len if ep_len > 0 else np.nan,
                # copia eixos para facilitar pivot depois
                "control": last_info.get("control", ""),
                "use_pi_reward": last_info.get("use_pi_reward", 0),
                "pi_metric": last_info.get("pi_metric", ""),
                "alpha_pi": last_info.get("alpha_pi", 0.0),
                "safety_filter": last_info.get("safety_filter", ""),
            }
        )

    env.close()

    summary = {
        "exp_id": exp_id,
        "train_seed": train_seed,
        "episodes": episodes,
        "eval_seed_base": eval_seed_base,
        "eval_seeds_range": [eval_seed_base, eval_seed_base + episodes - 1],
        "success_rate": succ / max(1, episodes),
        "mean_d0": float(np.nanmean(d0s)),
        "std_d0": float(np.nanstd(d0s)),
        "mean_final_dist": float(np.nanmean(final_dists)),
        "std_final_dist": float(np.nanstd(final_dists)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_len": float(np.mean(lengths)),
        "std_ep_len": float(np.std(lengths)),
        "mean_tau_effort": float(np.nanmean(tau_efforts)),
        "std_tau_effort": float(np.nanstd(tau_efforts)),
        "mean_pi_value": float(np.nanmean(pi_vals)),
        "std_pi_value": float(np.nanstd(pi_vals)),
        "mean_filter_intervention_rate": float(np.nanmean(filter_interventions)),
        "std_filter_intervention_rate": float(np.nanstd(filter_interventions)),
        "mean_filter_delta_tau_norm": float(np.nanmean(filter_delta_norms)),
        "std_filter_delta_tau_norm": float(np.nanstd(filter_delta_norms)),
        "mean_tau_raw_norm": (
            float(np.nanmean(tau_raw_norms)) if len(tau_raw_norms) else np.nan
        ),
        "mean_tau_cmd_norm": (
            float(np.nanmean(tau_cmd_norms)) if len(tau_cmd_norms) else np.nan
        ),
        "mean_tau_nom_norm": (
            float(np.nanmean(tau_nom_norms)) if len(tau_nom_norms) else np.nan
        ),
        "mean_tau_res_norm": (
            float(np.nanmean(tau_res_norms)) if len(tau_res_norms) else np.nan
        ),
    }

    return summary, rows


def save_outputs(
    exp_id: str,
    train_seed: int,
    eval_seed_base: int,
    rows: list,
    summary: dict,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(
        out_dir, f"eval__{exp_id}__trainseed{train_seed}__evalbase{eval_seed_base}.csv"
    )
    out_json = os.path.join(
        out_dir,
        f"eval__{exp_id}__trainseed{train_seed}__evalbase{eval_seed_base}__summary.json",
    )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_csv, out_json


def list_exp_ids():
    root = "runs_torque"
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        if os.path.isfile(os.path.join(root, name, "manifest.json")):
            out.append(name)
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-id",
        type=str,
        default="all",
        help="'all' ou o exp_id exato (pasta em runs_torque)",
    )
    ap.add_argument("--train-seed", type=int, default=0)

    ap.add_argument("--eval-seed-base", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=200)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--print-episodes", action="store_true")

    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    exp_ids = list_exp_ids() if args.exp_id == "all" else [args.exp_id]
    if not exp_ids:
        raise SystemExit(
            "Nenhum exp_id encontrado em runs_torque/ (sem manifest.json)."
        )

    all_summaries = []
    for exp_id in exp_ids:
        summary, rows = evaluate(
            exp_id=exp_id,
            train_seed=args.train_seed,
            eval_seed_base=args.eval_seed_base,
            episodes=args.episodes,
            render=args.render,
            print_episodes=args.print_episodes,
        )
        out_csv, out_json = save_outputs(
            exp_id, args.train_seed, args.eval_seed_base, rows, summary, args.out_dir
        )
        all_summaries.append(summary)
        print(
            f"[OK] {exp_id} -> {out_csv} | success_rate={summary['success_rate']:.3f}"
        )

    # salva índice geral das summaries
    out_all = os.path.join(
        args.out_dir,
        f"eval__ALL__trainseed{args.train_seed}__evalbase{args.eval_seed_base}__summaries.json",
    )
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary agregado: {out_all}")


if __name__ == "__main__":
    main()
