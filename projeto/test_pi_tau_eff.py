# test_pi_tau_eff.py
import argparse
import numpy as np

from two_link_arm_env import TwoLinkArmEnv


def run_episode(env: TwoLinkArmEnv, seed: int, steps: int, mode: str):
    """
    mode:
      - "random": ações aleatórias no action_space
      - "fixed": ação fixa não-zero (força o tau_cmd)
    """
    obs = env.reset(seed=seed)

    rows = []
    for t in range(steps):
        if mode == "random":
            action = env.action_space.sample()
        else:
            # ação fixa não-zero (garante que tau_cmd != 0 depois do clipping)
            action = np.array([0.6 * env.tau_max, -0.4 * env.tau_max], dtype=np.float32)

        obs, reward, done, info = env.step(action)

        rows.append(
            {
                "t": t,
                "reward": float(reward),
                "tau_cmd": (info.get("tau_cmd1", np.nan), info.get("tau_cmd2", np.nan)),
                "tau_app": (info.get("tau_app1", np.nan), info.get("tau_app2", np.nan)),
                "tau_eff": (info.get("tau_eff1", np.nan), info.get("tau_eff2", np.nan)),
                "tau_eff_source": info.get("tau_eff_source", None),
                "pi_value": info.get("pi_value", np.nan),
                "r_pi": info.get("r_pi", np.nan),
                "r_dist": info.get("r_dist", np.nan),
                "r_act": info.get("r_act", np.nan),
                "dist": info.get("distance", np.nan),
                "done": bool(done),
            }
        )

        if done:
            break

    return rows


def summarize(rows, label: str):
    tau_cmd = np.array([r["tau_cmd"] for r in rows], dtype=float)
    tau_app = np.array([r["tau_app"] for r in rows], dtype=float)
    tau_eff = np.array([r["tau_eff"] for r in rows], dtype=float)
    pi_value = np.array([r["pi_value"] for r in rows], dtype=float)
    r_pi = np.array([r["r_pi"] for r in rows], dtype=float)

    srcs = [r["tau_eff_source"] for r in rows]
    src_cmd = sum(1 for s in srcs if s == "cmd")
    src_applied = sum(1 for s in srcs if s == "applied")

    print("\n==============================")
    print(f"Resumo: {label}")
    print(f"steps: {len(rows)}")
    print(f"tau_cmd L2 mean: {np.mean(np.linalg.norm(tau_cmd, axis=1)):.6f}")
    print(f"tau_app L2 mean: {np.mean(np.linalg.norm(tau_app, axis=1)):.6f}")
    print(f"tau_eff L2 mean: {np.mean(np.linalg.norm(tau_eff, axis=1)):.6f}")
    print(
        f"pi_value mean/min/max: {np.mean(pi_value):.6f} / {np.min(pi_value):.6f} / {np.max(pi_value):.6f}"
    )
    print(
        f"r_pi mean/min/max:     {np.mean(r_pi):.6f} / {np.min(r_pi):.6f} / {np.max(r_pi):.6f}"
    )
    print(f"tau_eff_source counts: cmd={src_cmd}, applied={src_applied}")

    # Assert "soft" (não quebra, só avisa)
    if np.allclose(pi_value, 0.0, atol=1e-9):
        print("⚠️  ALERTA: pi_value ficou zerado em todos os passos.")
    if np.allclose(r_pi, 0.0, atol=1e-9):
        print("⚠️  ALERTA: r_pi ficou zerado em todos os passos.")
    if src_cmd == 0 and np.mean(np.linalg.norm(tau_cmd, axis=1)) > 1e-6:
        print(
            "⚠️  ALERTA: tau_cmd não-zero mas nunca caiu no fallback 'cmd' (talvez applied esteja ok)."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mode", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--pi-metric", choices=["tau_l1", "power"], default="tau_l1")
    ap.add_argument("--alpha-pi", type=float, default=5e-4)
    ap.add_argument("--tau-max", type=float, default=20.0)
    ap.add_argument("--control", choices=["direct", "residual"], default="direct")
    ap.add_argument(
        "--sf", choices=["none", "proj_box", "proj_box_jointlimit"], default="none"
    )
    args = ap.parse_args()

    # TESTE A: use_pi_reward=0 (r_pi deve ser 0)
    envA = TwoLinkArmEnv(
        render=False,
        control=args.control,
        use_pi_reward=False,
        pi_metric=args.pi_metric,
        alpha_pi=args.alpha_pi,
        safety_filter=args.sf,
        tau_max=args.tau_max,
    )
    rowsA = run_episode(envA, seed=args.seed, steps=args.steps, mode=args.mode)
    summarize(rowsA, "TESTE A (use_pi_reward=0)")
    envA.close()

    # TESTE B: use_pi_reward=1 (pi_value e r_pi devem ficar != 0 com ação fixa)
    envB = TwoLinkArmEnv(
        render=False,
        control=args.control,
        use_pi_reward=True,
        pi_metric=args.pi_metric,
        alpha_pi=args.alpha_pi,
        safety_filter=args.sf,
        tau_max=args.tau_max,
    )
    rowsB = run_episode(envB, seed=args.seed, steps=args.steps, mode=args.mode)
    summarize(rowsB, "TESTE B (use_pi_reward=1)")
    envB.close()

    # imprime 5 primeiras linhas do TESTE B pra inspeção rápida
    print("\nPrimeiros 5 passos do TESTE B:")
    for r in rowsB[:5]:
        print(
            f"t={r['t']:02d} "
            f"tau_cmd={r['tau_cmd']} tau_app={r['tau_app']} "
            f"tau_eff={r['tau_eff']} src={r['tau_eff_source']} "
            f"pi={r['pi_value']:.6f} r_pi={r['r_pi']:.6f} "
            f"reward={r['reward']:.6f}"
        )


if __name__ == "__main__":
    main()
