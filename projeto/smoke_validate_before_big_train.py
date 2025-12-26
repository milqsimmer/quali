# smoke_validate_before_big_train.py
import argparse
import sys
import numpy as np
import pybullet as p

from two_link_arm_env import TwoLinkArmEnv


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def make_env(**kwargs):
    # defaults seguros (iguais ao seu env atual)
    return TwoLinkArmEnv(
        render=False,
        seed=kwargs.pop("seed", 0),
        # sampling
        margin=kwargs.pop("margin", 0.02),
        min_tip_dist=kwargs.pop("min_tip_dist", 0.07),
        phi_min=kwargs.pop("phi_min", -np.pi / 2),
        phi_max=kwargs.pop("phi_max", +np.pi / 2),
        # success
        success_tol=kwargs.pop("success_tol", 0.05),
        # torques / reward
        tau_max=kwargs.pop("tau_max", 20.0),
        lam_a=kwargs.pop("lam_a", 0.001),
        # PI
        use_pi_reward=kwargs.pop("use_pi_reward", False),
        pi_metric=kwargs.pop("pi_metric", "tau_l1"),
        alpha_pi=kwargs.pop("alpha_pi", 0.0005),
        # safety
        safety_filter=kwargs.pop("safety_filter", "none"),
        dtau_max=kwargs.pop("dtau_max", 2.0),
        q_margin=kwargs.pop("q_margin", 0.15),
        # control axis
        control=kwargs.pop("control", "direct"),  # "direct" | "residual"
        kp=kwargs.pop("kp", 10.0),
        kd=kwargs.pop("kd", 1.0),
        elbow=kwargs.pop("elbow", "auto"),
    )


def check_seed_repro(seed_env=0, seed_reset=123):
    env = make_env(seed=seed_env)
    try:
        obs1 = env.reset(seed=seed_reset)
        t1 = np.array(env.target_pos, dtype=float)

        obs2 = env.reset(seed=seed_reset)
        t2 = np.array(env.target_pos, dtype=float)

        assert_true(
            np.allclose(t1, t2), f"[seed] target_pos não reproduziu: {t1} vs {t2}"
        )
        assert_true(np.allclose(obs1, obs2), "[seed] obs inicial não reproduziu")

        obs3 = env.reset(seed=seed_reset + 1)
        t3 = np.array(env.target_pos, dtype=float)
        assert_true(
            not np.allclose(t1, t3),
            "[seed] seed diferente gerou o mesmo target_pos (muito improvável)",
        )
    finally:
        env.close()


def check_gym_contract_and_obs_space(steps=20):
    env = make_env()
    try:
        obs = env.reset(seed=1000)
        assert_true(
            env.observation_space.contains(obs), f"[obs] reset fora do space: {obs}"
        )
        assert_true(np.all(np.isfinite(obs)), "[obs] reset tem NaN/inf")

        for _ in range(steps):
            a = env.action_space.sample()
            obs, r, done, info = env.step(a)
            assert_true(isinstance(done, (bool, np.bool_)), "[gym] done não é bool")
            assert_true(np.all(np.isfinite(obs)), "[obs] step tem NaN/inf")
            assert_true(
                env.observation_space.contains(obs), f"[obs] step fora do space: {obs}"
            )
            assert_true(np.isfinite(float(r)), "[reward] reward NaN/inf")
            if done:
                obs = env.reset(seed=1000)
    finally:
        env.close()


def one_step(env, action, seed=1000):
    env.reset(seed=seed)
    obs2, reward, done, info = env.step(np.array(action, dtype=np.float32))
    return obs2, float(reward), bool(done), info


def check_action_semantics_direct():
    env = make_env(control="direct", safety_filter="none", tau_max=20.0)
    try:
        _, _, _, info = one_step(env, action=[1000.0, -1000.0], seed=1001)

        # no seu env: action é clipada antes de virar tau_raw
        assert_true(
            abs(float(info["tau_raw1"])) <= 20.0 + 1e-6, "[action] tau_raw1 não clipou"
        )
        assert_true(
            abs(float(info["tau_raw2"])) <= 20.0 + 1e-6, "[action] tau_raw2 não clipou"
        )

        # safety_filter=none => tau_cmd == tau_raw
        raw = np.array([info["tau_raw1"], info["tau_raw2"]], dtype=float)
        cmd = np.array([info["tau_cmd1"], info["tau_cmd2"]], dtype=float)
        assert_true(
            np.allclose(raw, cmd, atol=1e-7), "[action] sf=none mas tau_cmd != tau_raw"
        )
    finally:
        env.close()


def check_tau_clipping_and_rate_limit():
    # força rate limit intervir
    env = make_env(
        control="direct", safety_filter="proj_box", tau_max=20.0, dtau_max=0.5
    )
    try:
        env.reset(seed=1000)

        # 1o passo: prev_tau=0 => cmd deve ficar em torno de +-0.5
        _, _, _, info1 = env.step(np.array([20.0, -20.0], dtype=np.float32))
        cmd1 = np.array([info1["tau_cmd1"], info1["tau_cmd2"]], dtype=float)
        assert_true(
            np.all(np.abs(cmd1) <= 0.5 + 1e-6),
            f"[rate] 1o passo não respeitou dtau_max: {cmd1}",
        )

        # 2o passo: delta <= 0.5 em relação ao cmd anterior
        _, _, _, info2 = env.step(np.array([20.0, -20.0], dtype=np.float32))
        cmd2 = np.array([info2["tau_cmd1"], info2["tau_cmd2"]], dtype=float)
        delta = cmd2 - cmd1
        assert_true(
            np.all(np.abs(delta) <= 0.5 + 1e-6),
            f"[rate] delta > dtau_max: delta={delta}",
        )

        # magnitude sempre respeitada
        assert_true(
            np.all(np.abs(cmd2) <= 20.0 + 1e-6), f"[tau] cmd fora de tau_max: {cmd2}"
        )

        # tau_eff coerente com a fonte indicada
        eff = np.array([info2["tau_eff1"], info2["tau_eff2"]], dtype=float)
        src = info2.get("tau_eff_source", "")
        if src == "cmd":
            assert_true(
                np.allclose(eff, cmd2, atol=1e-6),
                "[tau_eff] source=cmd mas tau_eff != tau_cmd",
            )
        elif src == "applied":
            app = np.array([info2["tau_app1"], info2["tau_app2"]], dtype=float)
            assert_true(
                np.allclose(eff, app, atol=1e-6),
                "[tau_eff] source=applied mas tau_eff != tau_app",
            )
        else:
            assert_true(False, f"[tau_eff] tau_eff_source inesperado: {src}")
    finally:
        env.close()


def check_reward_identity(use_pi_reward: bool):
    env = make_env(
        control="direct",
        safety_filter="none",
        use_pi_reward=use_pi_reward,
        alpha_pi=0.0005,
    )
    try:
        _, reward, _, info = one_step(env, action=[5.0, -3.0], seed=1002)

        r_dist = float(info.get("r_dist", 0.0))
        r_act = float(info.get("r_act", 0.0))
        r_pi = float(info.get("r_pi", 0.0))
        pi_val = float(info.get("pi_value", 0.0))

        assert_true(
            abs(reward - (r_dist + r_act + r_pi)) <= 1e-6,
            f"[reward] reward != soma: reward={reward} soma={r_dist+r_act+r_pi}",
        )

        if not use_pi_reward:
            assert_true(abs(r_pi) <= 1e-12, f"[pi] use_pi_reward=0 mas r_pi={r_pi}")
            assert_true(
                abs(pi_val) <= 1e-12, f"[pi] use_pi_reward=0 mas pi_value={pi_val}"
            )
        else:
            assert_true(pi_val > 0.0, f"[pi] use_pi_reward=1 mas pi_value={pi_val}")
            assert_true(
                r_pi < 0.0, f"[pi] use_pi_reward=1 mas r_pi não é negativo: {r_pi}"
            )
    finally:
        env.close()


def check_safety_filter_intervenes():
    env = make_env(
        control="direct",
        safety_filter="proj_box_jointlimit",
        dtau_max=0.5,
        q_margin=0.15,
    )
    try:
        env.reset(seed=1000)

        intervened = 0
        max_delta = 0.0

        for _ in range(10):
            _, _, _, info = env.step(np.array([20.0, -20.0], dtype=np.float32))
            intervened += int(info.get("filter_intervened", 0))
            max_delta = max(max_delta, float(info.get("filter_delta_tau_norm", 0.0)))

            if int(info.get("filter_intervened", 0)) == 1:
                raw = np.array(
                    [info.get("tau_raw1", 0.0), info.get("tau_raw2", 0.0)], float
                )
                cmd = np.array(
                    [info.get("tau_cmd1", 0.0), info.get("tau_cmd2", 0.0)], float
                )
                assert_true(
                    not np.allclose(raw, cmd),
                    "[sf] intervened=1 mas tau_raw == tau_cmd",
                )

        assert_true(
            intervened > 0,
            "[sf] safety filter nunca interveio em 10 passos com ação grande",
        )
        assert_true(max_delta > 0.0, "[sf] filter_delta_tau_norm nunca > 0")
    finally:
        env.close()


def check_residual_identities():
    env = make_env(control="residual", safety_filter="none", kp=10.0, kd=1.0)
    try:
        _, _, _, info = one_step(env, action=[5.0, -5.0], seed=1003)

        tau_nom = np.array(
            [info.get("tau_nom1", np.nan), info.get("tau_nom2", np.nan)], float
        )
        tau_res = np.array(
            [info.get("tau_res1", np.nan), info.get("tau_res2", np.nan)], float
        )
        tau_raw = np.array(
            [info.get("tau_raw1", np.nan), info.get("tau_raw2", np.nan)], float
        )
        tau_cmd = np.array(
            [info.get("tau_cmd1", np.nan), info.get("tau_cmd2", np.nan)], float
        )

        assert_true(np.all(np.isfinite(tau_nom)), "[residual] tau_nom NaN/inf")
        assert_true(np.all(np.isfinite(tau_res)), "[residual] tau_res NaN/inf")
        assert_true(np.all(np.isfinite(tau_raw)), "[residual] tau_raw NaN/inf")

        # sf=none => cmd==raw e raw==nom+res
        assert_true(
            np.allclose(tau_cmd, tau_raw, atol=1e-7),
            "[residual] sf=none mas tau_cmd != tau_raw",
        )
        assert_true(
            np.allclose(tau_raw, tau_nom + tau_res, atol=1e-6),
            "[residual] tau_raw != tau_nom + tau_res",
        )
    finally:
        env.close()


def check_tau_l1_nonzero(steps=50):
    env = make_env(control="direct", safety_filter="none", use_pi_reward=True)
    try:
        env.reset(seed=1000)
        s = 0.0
        for _ in range(steps):
            a = env.action_space.sample()
            _, _, done, info = env.step(a)
            s += float(info.get("tau_l1", 0.0))
            if done:
                env.reset(seed=1000)

        assert_true(
            s > 0.0,
            "[metrics] soma de tau_l1 ficou 0 (tau_l1 não está sendo preenchido?)",
        )
    finally:
        env.close()


def check_determinism_same_actions(seed_env=0, seed_reset=1234, n_steps=15):
    """
    Mesmo env + mesmo reset seed + mesma sequência de ações =>
    mesma trajetória (obs/reward/info) até tolerância.
    """
    env = make_env(
        seed=seed_env, control="direct", safety_filter="none", use_pi_reward=True
    )

    try:
        rng = np.random.default_rng(2025)
        actions = rng.uniform(-env.tau_max, env.tau_max, size=(n_steps, 2)).astype(
            np.float32
        )

        def rollout():
            obs0 = env.reset(seed=seed_reset)
            traj = []
            for k in range(n_steps):
                o, r, d, info = env.step(actions[k])
                traj.append(
                    (
                        np.array(o, dtype=float),
                        float(r),
                        bool(d),
                        {  # pega só campos relevantes e estáveis
                            "distance": float(info.get("distance", np.nan)),
                            "tau_cmd1": float(info.get("tau_cmd1", np.nan)),
                            "tau_cmd2": float(info.get("tau_cmd2", np.nan)),
                            "tau_eff1": float(info.get("tau_eff1", np.nan)),
                            "tau_eff2": float(info.get("tau_eff2", np.nan)),
                            "pi_value": float(info.get("pi_value", np.nan)),
                            "tau_l1": float(info.get("tau_l1", np.nan)),
                        },
                    )
                )
                if d:
                    break
            return np.array(obs0, dtype=float), traj

        obs_a, traj_a = rollout()
        obs_b, traj_b = rollout()

        assert_true(
            np.allclose(obs_a, obs_b, atol=1e-7),
            "[det] obs reset divergiu no mesmo env",
        )
        assert_true(
            len(traj_a) == len(traj_b), "[det] rollout com comprimentos diferentes"
        )

        for k, (A, B) in enumerate(zip(traj_a, traj_b)):
            o1, r1, d1, i1 = A
            o2, r2, d2, i2 = B
            assert_true(
                np.allclose(o1, o2, atol=1e-6), f"[det] obs divergiu no step {k}"
            )
            assert_true(abs(r1 - r2) <= 1e-6, f"[det] reward divergiu no step {k}")
            assert_true(d1 == d2, f"[det] done divergiu no step {k}")
            for key in i1.keys():
                assert_true(
                    abs(i1[key] - i2[key]) <= 1e-6,
                    f"[det] info[{key}] divergiu no step {k}",
                )
    finally:
        env.close()


def check_jointlimit_blocks_push():
    """
    Coloca as juntas explicitamente perto de +/- pi e verifica que o filtro
    zera torque que empurra mais para fora.
    """
    env = make_env(
        control="direct",
        safety_filter="proj_box_jointlimit",
        dtau_max=999.0,
        q_margin=0.15,
    )
    try:
        env.reset(seed=1000)

        # força estado perto do limite
        q0 = np.pi - env.q_margin / 2.0
        q1 = -np.pi + env.q_margin / 2.0

        p.resetJointState(
            env.robot, env.joint_ids[0], targetValue=float(q0), targetVelocity=0.0
        )
        p.resetJointState(
            env.robot, env.joint_ids[1], targetValue=float(q1), targetVelocity=0.0
        )
        p.stepSimulation()
        env._read_joint_state(return_applied_tau=True)  # atualiza env.theta1/theta2

        # aplica torque que empurra para fora (deveria ser bloqueado)
        _, _, _, info = env.step(np.array([+5.0, -5.0], dtype=np.float32))

        tau_cmd1 = float(info.get("tau_cmd1", np.nan))
        tau_cmd2 = float(info.get("tau_cmd2", np.nan))

        assert_true(
            abs(tau_cmd1) <= 1e-9,
            f"[jointlimit] deveria zerar tau_cmd1 perto de +pi, mas deu {tau_cmd1}",
        )
        assert_true(
            abs(tau_cmd2) <= 1e-9,
            f"[jointlimit] deveria zerar tau_cmd2 perto de -pi, mas deu {tau_cmd2}",
        )

        assert_true(
            int(info.get("filter_intervened", 0)) == 1,
            "[jointlimit] deveria marcar filter_intervened=1",
        )
        assert_true(
            float(info.get("filter_delta_tau_norm", 0.0)) > 0.0,
            "[jointlimit] delta_tau_norm deveria ser > 0",
        )

    finally:
        env.close()


def check_mini_rollout_sanity(episodes=2, max_steps=200):
    """
    Mini rollout sem SB3 só para pegar bugs de logging / métricas NaN
    e coerência geral dos infos.
    """
    env = make_env(
        control="direct", safety_filter="proj_box", use_pi_reward=True, dtau_max=2.0
    )
    try:
        total_steps = 0
        pi_vals = []
        tau_l1_vals = []
        dists = []

        for ep in range(episodes):
            env.reset(seed=1000 + ep)
            for _ in range(max_steps):
                a = env.action_space.sample()
                obs, r, done, info = env.step(a)

                assert_true(np.all(np.isfinite(obs)), "[mini] obs NaN/inf")
                assert_true(np.isfinite(float(r)), "[mini] reward NaN/inf")

                pi_vals.append(float(info.get("pi_value", np.nan)))
                tau_l1_vals.append(float(info.get("tau_l1", np.nan)))
                dists.append(float(info.get("distance", np.nan)))

                total_steps += 1
                if done:
                    break

        assert_true(total_steps > 0, "[mini] nenhum step rodou")
        assert_true(np.all(np.isfinite(pi_vals)), "[mini] pi_value com NaN/inf")
        assert_true(np.all(np.isfinite(tau_l1_vals)), "[mini] tau_l1 com NaN/inf")
        assert_true(np.all(np.isfinite(dists)), "[mini] distance com NaN/inf")

        assert_true(np.mean(tau_l1_vals) > 0.0, "[mini] mean tau_l1 ficou 0 (estranho)")
        assert_true(
            np.mean(pi_vals) > 0.0,
            "[mini] mean pi_value ficou 0 com use_pi_reward=True (estranho)",
        )

    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    tests = [
        ("seed_repro", lambda: check_seed_repro(seed_env=args.seed, seed_reset=123)),
        ("gym_contract_obs_space", lambda: check_gym_contract_and_obs_space(steps=20)),
        ("action_semantics_direct", check_action_semantics_direct),
        ("tau_clip_rate_limit", check_tau_clipping_and_rate_limit),
        ("reward_identity_pi0", lambda: check_reward_identity(use_pi_reward=False)),
        ("reward_identity_pi1", lambda: check_reward_identity(use_pi_reward=True)),
        ("safety_filter", check_safety_filter_intervenes),
        ("residual_identities", check_residual_identities),
        ("tau_l1_nonzero", lambda: check_tau_l1_nonzero(steps=args.steps)),
        (
            "determinism_same_actions",
            lambda: check_determinism_same_actions(seed_env=args.seed),
        ),
        ("jointlimit_blocks_push", check_jointlimit_blocks_push),
        ("mini_rollout_sanity", check_mini_rollout_sanity),
    ]

    print("=== SMOKE VALIDATION (before big train) ===")
    ok = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[OK] {name}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            sys.exit(1)

    print(f"\nALL PASS ({ok}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
