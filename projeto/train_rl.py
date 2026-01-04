# train_rl.py (TORQUE + matriz completa: control / PI / safety)
import argparse
import os
import json

from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from two_link_arm_env import TwoLinkArmEnv


def build_exp_id(args) -> str:
    parts = []
    parts.append(f"ctrl-{args.control}")
    parts.append(
        f"pi-{int(args.use_pi_reward)}-{args.pi_metric}-a{args.alpha_pi:g}"
        f"-pg{args.pi_gating}-d{args.pi_gate_dist:g}-dm{(args.pi_gate_min_dist if args.pi_gate_min_dist is not None else 'st')}"
    )
    parts.append(f"sf-{args.safety_filter}-dt{args.dtau_max:g}-qm{args.q_margin:g}")
    if args.control == "residual":
        parts.append(f"pd-kp{args.kp:g}-kd{args.kd:g}-el{args.elbow}")
    parts.append(f"tau{args.tau_max:g}-lama{args.lam_a:g}")
    parts.append(
        "rew-"
        f"{args.task_reward}"
        f"-sb{args.success_bonus:g}"
        f"-lt{args.lam_time:g}"
        f"-lv{args.lam_v:g}"
        f"-ls{args.lam_smooth:g}"
        f"-lq{args.lam_q:g}"
        f"-ek{args.exp_k:g}"
    )
    parts.append(
        f"a2-m{args.margin:g}-md{args.min_tip_dist:g}-phi{args.phi_min:g}_{args.phi_max:g}"
    )
    parts.append(f"h{args.max_steps}")
    return "__".join(parts)


def main():
    ap = argparse.ArgumentParser()

    # experiment axes
    ap.add_argument("--control", choices=["direct", "residual"], default="direct")
    ap.add_argument("--use-pi-reward", action="store_true")
    ap.add_argument("--pi-metric", choices=["tau_l1", "power"], default="tau_l1")
    ap.add_argument("--alpha-pi", type=float, default=0.0005)

    # PI gating (distance-based)
    ap.add_argument("--pi-gating", choices=["none", "distance"], default="none")
    ap.add_argument("--pi-gate-dist", type=float, default=0.25)
    ap.add_argument("--pi-gate-min-dist", type=float, default=None)

    ap.add_argument(
        "--safety-filter",
        choices=["none", "proj_box", "proj_box_jointlimit"],
        default="none",
    )
    ap.add_argument("--dtau-max", type=float, default=2.0)
    ap.add_argument("--q-margin", type=float, default=0.15)

    # residual nominal PD
    ap.add_argument("--kp", type=float, default=10.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--elbow", choices=["auto", "up", "down"], default="auto")

    # env basic
    ap.add_argument("--tau-max", type=float, default=20.0)
    ap.add_argument("--lam-a", type=float, default=0.001)

    # reward shaping
    ap.add_argument(
        "--task-reward", type=str, default="dist", choices=["dist", "progress", "exp"]
    )
    ap.add_argument("--exp-k", type=float, default=5.0)
    ap.add_argument("--lam-time", type=float, default=0.0)
    ap.add_argument("--lam-v", type=float, default=0.0)
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--lam-q", type=float, default=0.0)
    ap.add_argument("--success-bonus", type=float, default=0.0)

    ap.add_argument("--success-tol", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)

    # A2 sampler
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-tip-dist", type=float, default=0.07)
    ap.add_argument("--phi-min", type=float, default=-3.141592653589793 / 2)
    ap.add_argument("--phi-max", type=float, default=+3.141592653589793 / 2)

    # training
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)

    # (opcional) sobrescrever exp_id
    ap.add_argument("--exp-id", type=str, default="")

    args = ap.parse_args()

    exp_id = args.exp_id.strip() or build_exp_id(args)

    env = TwoLinkArmEnv(
        render=False,
        seed=args.seed,
        # A2
        margin=args.margin,
        min_tip_dist=args.min_tip_dist,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        # episode
        success_tol=args.success_tol,
        # torque
        tau_max=args.tau_max,
        # reward
        lam_a=args.lam_a,
        task_reward=args.task_reward,
        exp_k=args.exp_k,
        lam_time=args.lam_time,
        lam_v=args.lam_v,
        lam_smooth=args.lam_smooth,
        lam_q=args.lam_q,
        success_bonus=args.success_bonus,
        # PI
        use_pi_reward=args.use_pi_reward,
        pi_metric=args.pi_metric,
        alpha_pi=args.alpha_pi,
        # PI gating
        pi_gating=args.pi_gating,
        pi_gate_dist=args.pi_gate_dist,
        pi_gate_min_dist=args.pi_gate_min_dist,
        # safety
        safety_filter=args.safety_filter,
        dtau_max=args.dtau_max,
        q_margin=args.q_margin,
        # residual
        control=args.control,
        kp=args.kp,
        kd=args.kd,
        elbow=args.elbow,
    )

    env = TimeLimit(env, max_episode_steps=args.max_steps)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    model.learn(total_timesteps=args.steps)

    out_dir = os.path.join("runs_torque", exp_id)
    os.makedirs(out_dir, exist_ok=True)

    # salva modelo
    model_path = os.path.join(out_dir, f"ppo_model_seed{args.seed}.zip")
    model.save(model_path)

    # salva manifest (parâmetros do experimento + hiperparâmetros SB3)
    manifest = {
        "exp_id": exp_id,
        "env": {
            "control": args.control,
            "use_pi_reward": bool(args.use_pi_reward),
            "pi_metric": args.pi_metric,
            "alpha_pi": args.alpha_pi,
            "pi_gating": args.pi_gating,
            "pi_gate_dist": args.pi_gate_dist,
            "pi_gate_min_dist": args.pi_gate_min_dist,
            "safety_filter": args.safety_filter,
            "dtau_max": args.dtau_max,
            "q_margin": args.q_margin,
            "kp": args.kp,
            "kd": args.kd,
            "elbow": args.elbow,
            "tau_max": args.tau_max,
            "lam_a": args.lam_a,
            "task_reward": args.task_reward,
            "exp_k": args.exp_k,
            "lam_time": args.lam_time,
            "lam_v": args.lam_v,
            "lam_smooth": args.lam_smooth,
            "lam_q": args.lam_q,
            "success_bonus": args.success_bonus,
            "success_tol": args.success_tol,
            "margin": args.margin,
            "min_tip_dist": args.min_tip_dist,
            "phi_min": args.phi_min,
            "phi_max": args.phi_max,
            "max_steps": args.max_steps,
        },
        "train": {
            "steps": args.steps,
            "seed": args.seed,
            "algo": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        "artifacts": {
            "model_path": model_path,
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    env.close()
    print(f"[OK] Treino finalizado.\nexp_id={exp_id}\nmodelo={model_path}")


if __name__ == "__main__":
    main()
