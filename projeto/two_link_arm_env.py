# two_link_arm_env.py  (TORQUE_CONTROL + PI-reward + Safety filter + Residual RL)

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


def _wrap_to_pi(x: float) -> float:
    # coloca erro angular em [-pi, pi]
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class TwoLinkArmEnv(gym.Env):
    """
    2-DoF planar arm (PyBullet) with direct torque control.

    Eixos (proposta completa):
      - control: "direct" or "residual"
      - PI reward: use_pi_reward (bool), pi_metric ("tau_l1" or "power"), alpha_pi
      - safety_filter: "none", "proj_box", "proj_box_jointlimit"
        * proj_box = projeção (L2) no conjunto: |tau|<=tau_max e |tau - tau_prev|<=dtau_max
        * jointlimit = além disso, bloqueia torque que empurra ainda mais para fora quando perto do limite articular
      - residual nominal: PD em juntas para rastrear q_des (via IK analítico do alvo)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render=False,
        seed=None,
        # ----- geometria -----
        l1=0.5,
        l2=0.5,
        # ----- alvo -----
        margin=0.02,
        min_tip_dist=0.07,
        phi_min=-np.pi / 2,
        phi_max=+np.pi / 2,
        # ----- episodio -----
        success_tol=0.05,
        # ----- torque -----
        tau_max=20.0,
        # ----- reward (task + regularizações) -----
        # Defaults preservam o comportamento anterior (task_reward="dist" e coeficientes = 0).
        lam_a=0.001,  # esforço (torque)
        task_reward="dist",  # "dist" | "progress" | "exp"
        exp_k=5.0,  # usado quando task_reward == "exp"
        lam_time=0.0,  # penalidade por passo (incentiva chegar mais rápido)
        lam_v=0.0,  # penalidade de velocidade articular (qdot)
        lam_smooth=0.0,  # penalidade de variação de torque (|tau_t - tau_{t-1}|)
        lam_q=0.0,  # penalidade por proximidade/violação de limite articular (soft)
        success_bonus=0.0,  # bônus ao atingir o alvo
        # ----- PI-reward -----
        use_pi_reward=False,
        pi_metric="tau_l1",  # "tau_l1" ou "power"
        alpha_pi=0.0005,
        # ----- PI-reward gating (distance-based) -----
        pi_gating="none",  # "none" | "distance"
        pi_gate_dist=0.25,  # começa a ligar PI quando dist <= pi_gate_dist
        pi_gate_min_dist=None,  # onde PI fica 100% (default = success_tol)
        # ----- safety filter -----
        safety_filter="none",  # "none", "proj_box", "proj_box_jointlimit"
        dtau_max=2.0,  # rate limit por passo (N·m)
        q_margin=0.15,  # margem perto do limite articular (rad)
        # ----- residual RL -----
        control="direct",  # "direct" ou "residual"
        kp=10.0,
        kd=1.0,
        elbow="auto",  # "auto", "up", "down"
    ):
        super().__init__()

        assert control in ("direct", "residual")
        assert safety_filter in ("none", "proj_box", "proj_box_jointlimit")
        assert pi_metric in ("tau_l1", "power")
        assert task_reward in ("dist", "progress", "exp")

        self.render_mode = bool(render)

        # params geom
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.offset_local = [self.l2, 0, 0]

        # sampler
        self.margin = float(margin)
        self.min_tip_dist = float(min_tip_dist)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        # success
        self.success_tol = float(success_tol)

        # torque
        self.tau_max = float(tau_max)

        # reward
        self.lam_a = float(lam_a)
        self.task_reward = str(task_reward)
        self.exp_k = float(exp_k)
        self.lam_time = float(lam_time)
        self.lam_v = float(lam_v)
        self.lam_smooth = float(lam_smooth)
        self.lam_q = float(lam_q)
        self.success_bonus = float(success_bonus)

        # PI reward
        self.use_pi_reward = bool(use_pi_reward)
        self.pi_metric = pi_metric
        self.alpha_pi = float(alpha_pi)
        if self.use_pi_reward and self.lam_a > 0:
            print(
                f"[INFO] Ajustando lam_a de {self.lam_a} para 0.0 porque use_pi_reward=1 "
                f"(evitar dupla penalização de torque)."
            )
            self.lam_a = 0.0

        # PI gating
        assert pi_gating in ("none", "distance")
        self.pi_gating = str(pi_gating)
        self.pi_gate_dist = float(pi_gate_dist)
        self.pi_gate_min_dist = (
            None if (pi_gate_min_dist is None) else float(pi_gate_min_dist)
        )

        # safety
        self.safety_filter = safety_filter
        self.dtau_max = float(dtau_max)
        self.q_margin = float(q_margin)

        # residual
        self.control = control
        self.kp = float(kp)
        self.kd = float(kd)
        self.elbow = elbow

        # PyBullet init
        self.physicsClient = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

        self.joint_ids = [0, 1]
        self.link2 = 1

        # Spaces
        # obs: [theta1, theta2, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -1.5, -1.5], dtype=np.float32),
            high=np.array([np.pi, np.pi, 1.5, 1.5], dtype=np.float32),
            dtype=np.float32,
        )

        # action sempre 2D: no direct = tau; no residual = tau_res (mas ainda em N·m)
        self.action_space = spaces.Box(
            low=np.array([-self.tau_max, -self.tau_max], dtype=np.float32),
            high=np.array([self.tau_max, self.tau_max], dtype=np.float32),
            dtype=np.float32,
        )

        # RNG
        self.np_random, _ = seeding.np_random(seed)

        # state
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.qdot1 = 0.0
        self.qdot2 = 0.0

        self.target_pos = None
        self.d0 = None

        # residual nominal target in joint space (computed at reset)
        self.q_des = np.array([0.0, 0.0], dtype=float)

        # safety memory
        self.prev_tau_applied = np.array([0.0, 0.0], dtype=float)

        # reward shaping memory
        self.prev_dist = None  # set on reset
        self.prev_tau_cmd = np.array([0.0, 0.0], dtype=float)

        # enable torque mode
        self._disable_motors()

    # ---------------- Gym API ----------------

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        # reset joints to 0
        for j in self.joint_ids:
            p.resetJointState(self.robot, j, targetValue=0.0, targetVelocity=0.0)

        self._disable_motors()

        # apply 0 torque and step once
        self._apply_torque(np.array([0.0, 0.0], dtype=float))
        p.stepSimulation()

        # read real state
        self._read_joint_state()

        tip0 = np.array(self._get_end_effector_pos()[:2], dtype=float)

        # sample target
        self.target_pos = self._sample_target(tip0=tip0)
        self.d0 = float(np.linalg.norm(tip0 - np.array(self.target_pos, dtype=float)))

        # reward shaping memory
        self.prev_dist = float(self.d0)
        self.prev_tau_cmd = np.array([0.0, 0.0], dtype=float)

        # compute nominal joint target via IK (for residual)
        self.q_des = self._ik_2link(
            np.array(self.target_pos, dtype=float), elbow=self.elbow
        )

        # safety memory
        self.prev_tau_applied = np.array([0.0, 0.0], dtype=float)

        self._create_target_visual()
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            2,
        )
        action = np.clip(action, -self.tau_max, self.tau_max)

        # compute raw torque command
        if self.control == "direct":
            tau_raw = action.astype(float)
            tau_res = np.array([np.nan, np.nan], dtype=float)
            tau_nom = np.array([np.nan, np.nan], dtype=float)
        else:
            # residual: tau = tau_nom + tau_res
            tau_res = action.astype(float)
            tau_nom = self._nominal_pd_torque(q_des=self.q_des)
            tau_raw = tau_nom + tau_res

        # apply safety filter (if enabled)
        tau_cmd, filt_info = self._safety_filter(tau_raw)

        # apply torque & step
        self._apply_torque(tau_cmd)
        p.stepSimulation()

        # read real state and applied torque
        tau_applied, q, qdot = self._read_joint_state(return_applied_tau=True)

        # distance to target
        ee = self._get_end_effector_pos()
        dist = float(
            np.linalg.norm(
                np.array(ee[:2], dtype=float) - np.array(self.target_pos, dtype=float)
            )
        )

        # ---------- reward decomposition ----------
        # (1) Termo de tarefa (sempre guardamos r_dist para análise)
        r_dist = -dist
        r_prog = 0.0
        if self.prev_dist is not None:
            r_prog = float(self.prev_dist - dist)  # progresso >0 quando aproxima

        if self.task_reward == "dist":
            r_task = float(r_dist)
        elif self.task_reward == "progress":
            r_task = float(r_prog)
        else:  # "exp"
            # recompensa positiva e saturada perto do alvo
            r_task = float(np.exp(-self.exp_k * dist))

        # (2) Regularizações: torque (base), velocidade, suavidade, tempo
        r_act = -self.lam_a * float(np.linalg.norm(tau_cmd))
        r_vel = -self.lam_v * float(np.linalg.norm(qdot) ** 2)

        # suavidade: usa delta tau_cmd
        dtau_cmd = np.array(tau_cmd, dtype=float) - np.array(
            self.prev_tau_cmd, dtype=float
        )
        dtau_cmd_norm = float(np.linalg.norm(dtau_cmd))
        r_smooth = -self.lam_smooth * float(dtau_cmd_norm**2)

        r_time = -self.lam_time

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Robust torque choice for PI/métricas:
        # some setups return appliedMotorTorque ~ 0 even under TORQUE_CONTROL.
        tau_eff = np.array(tau_applied, dtype=float).copy()
        tau_eff_source = "applied"
        if np.allclose(tau_eff, 0.0, atol=1e-6) and (np.linalg.norm(tau_cmd) > 1e-6):
            tau_eff = np.array(tau_cmd, dtype=float).copy()
            tau_eff_source = "cmd"
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        tau_l1_eff = float(abs(tau_eff[0]) + abs(tau_eff[1]))

        # potência proxy (sempre calculada p/ métrica e logging)
        power_abs = float(abs(tau_eff[0] * qdot[0]) + abs(tau_eff[1] * qdot[1]))

        # (3) PI term (physics-informed): valor SEMPRE é calculado p/ métricas.
        if self.pi_metric == "tau_l1":
            pi_metric_val = tau_l1_eff
        else:  # "power"
            pi_metric_val = power_abs

            # (3) PI term (physics-informed): valor SEMPRE é calculado p/ métricas.
        if self.pi_metric == "tau_l1":
            pi_metric_val = tau_l1_eff
        else:  # "power"
            pi_metric_val = power_abs

        # gating por distância: PI só pesa perto do alvo
        # g in [0,1], onde g=0 (longe) e g=1 (muito perto)
        if self.pi_gating == "distance":
            d_gate = float(self.pi_gate_dist)
            d_min = float(
                self.success_tol
                if self.pi_gate_min_dist is None
                else self.pi_gate_min_dist
            )
            # segurança numérica e coerência
            if d_gate <= d_min:
                g_pi = 1.0 if dist <= d_gate else 0.0
            else:
                g_pi = float(np.clip((d_gate - dist) / (d_gate - d_min), 0.0, 1.0))
        else:
            g_pi = 1.0

        if self.use_pi_reward:
            pi_val = float(pi_metric_val)
            r_pi = -self.alpha_pi * pi_val * g_pi
            alpha_used = self.alpha_pi
        else:
            pi_val = 0.0
            r_pi = 0.0
            alpha_used = 0.0

        # (4) Penalidade suave por proximidade de limites articulares (opcional)
        q_abs = np.abs(np.array(q, dtype=float))
        q_soft = np.pi - float(self.q_margin)
        q_viol = np.maximum(0.0, q_abs - q_soft) / max(1e-9, float(self.q_margin))
        q_viol_norm2 = float(np.sum(q_viol**2))
        r_q = -self.lam_q * q_viol_norm2

        # (5) Bônus terminal
        done = dist <= self.success_tol
        r_success = float(self.success_bonus) if done else 0.0

        # total
        reward = float(
            r_task + r_act + r_pi + r_vel + r_smooth + r_time + r_q + r_success
        )

        # update shaping memory
        self.prev_dist = float(dist)
        self.prev_tau_cmd = np.array(tau_cmd, dtype=float)

        info = {
            "distance": dist,
            "success_tol": self.success_tol,
            "success": int(done),
            "d0": float(self.d0) if self.d0 is not None else np.nan,
            # method axes
            "control": self.control,
            "use_pi_reward": int(self.use_pi_reward),
            "pi_metric": self.pi_metric,
            "pi_gate": float(g_pi),
            "pi_gate_dist": float(self.pi_gate_dist),
            "pi_gate_min_dist": float(
                self.success_tol
                if self.pi_gate_min_dist is None
                else self.pi_gate_min_dist
            ),
            "pi_gating": self.pi_gating,
            "alpha_pi": float(alpha_used),
            "safety_filter": self.safety_filter,
            # reward config (for reproducibility)
            "task_reward": self.task_reward,
            "exp_k": float(self.exp_k),
            "lam_time": float(self.lam_time),
            "lam_v": float(self.lam_v),
            "lam_smooth": float(self.lam_smooth),
            "lam_q": float(self.lam_q),
            "success_bonus": float(self.success_bonus),
            # torques
            "tau_raw1": float(tau_raw[0]),
            "tau_raw2": float(tau_raw[1]),
            "tau_cmd1": float(tau_cmd[0]),
            "tau_cmd2": float(tau_cmd[1]),
            "tau_app1": float(tau_applied[0]),
            "tau_app2": float(tau_applied[1]),
            # effective torque + source
            "tau_eff1": float(tau_eff[0]),
            "tau_eff2": float(tau_eff[1]),
            "tau_eff_source": tau_eff_source,
            "tau_l1": tau_l1_eff,
            # velocities
            "qdot1": float(qdot[0]),
            "qdot2": float(qdot[1]),
            "qdot_norm": float(np.linalg.norm(qdot)),
            # residual details (when applicable)
            "tau_nom1": float(tau_nom[0]) if np.isfinite(tau_nom[0]) else np.nan,
            "tau_nom2": float(tau_nom[1]) if np.isfinite(tau_nom[1]) else np.nan,
            "tau_res1": float(tau_res[0]) if np.isfinite(tau_res[0]) else np.nan,
            "tau_res2": float(tau_res[1]) if np.isfinite(tau_res[1]) else np.nan,
            # reward components
            "r_dist": float(r_dist),
            "r_prog": float(r_prog),
            "r_task": float(r_task),
            "r_act": float(r_act),
            "r_pi": float(r_pi),
            "r_vel": float(r_vel),
            "r_smooth": float(r_smooth),
            "r_time": float(r_time),
            "r_q": float(r_q),
            "r_success": float(r_success),
            # PI values
            "pi_value": float(pi_val),
            "pi_value_metric": float(pi_metric_val),
            # extra metrics
            "power_abs": float(power_abs),
            "dtau_cmd_norm": float(dtau_cmd_norm),
            "q_viol_norm2": float(q_viol_norm2),
            # safety metrics
            **filt_info,
        }

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        if self.render_mode:
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect(self.physicsClient)

    # ---------------- internal helpers ----------------

    def _disable_motors(self):
        for j in self.joint_ids:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)

    def _apply_torque(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(
            2,
        )
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        p.setJointMotorControl2(self.robot, 0, p.TORQUE_CONTROL, force=float(tau[0]))
        p.setJointMotorControl2(self.robot, 1, p.TORQUE_CONTROL, force=float(tau[1]))

    def _read_joint_state(self, return_applied_tau=False):
        js = p.getJointStates(self.robot, self.joint_ids)
        q = np.array([float(js[0][0]), float(js[1][0])], dtype=float)
        qdot = np.array([float(js[0][1]), float(js[1][1])], dtype=float)
        tau_applied = np.array([float(js[0][3]), float(js[1][3])], dtype=float)

        self.theta1, self.theta2 = q[0], q[1]
        self.qdot1, self.qdot2 = qdot[0], qdot[1]

        if return_applied_tau:
            return tau_applied, q, qdot
        return q, qdot

    def _get_obs(self):
        return np.array(
            [
                float(self.theta1),
                float(self.theta2),
                float(self.target_pos[0]),
                float(self.target_pos[1]),
            ],
            dtype=np.float32,
        )

    def _create_target_visual(self):
        if hasattr(self, "target_id"):
            p.removeBody(self.target_id)

        radius = 0.03
        red = [1, 0, 0, 1]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=red)
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=vis,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.1],
        )

    def _get_end_effector_pos(self):
        ls = p.getLinkState(self.robot, self.link2, computeForwardKinematics=True)
        link_pos = ls[4] if len(ls) > 4 else ls[0]
        link_ornt = ls[5] if len(ls) > 5 else ls[1]
        ee_pos, _ = p.multiplyTransforms(
            link_pos, link_ornt, self.offset_local, [0, 0, 0, 1]
        )
        return ee_pos

    # ---------------- A2 sampler ----------------

    def _sample_target(self, tip0=None):
        r_min = abs(self.l1 - self.l2) + self.margin
        r_max = (self.l1 + self.l2) - self.margin

        if tip0 is None:
            tip0 = np.array([self.l1 + self.l2, 0.0], dtype=float)
        else:
            tip0 = np.array(tip0, dtype=float)

        while True:
            u = self.np_random.uniform(0.0, 1.0)
            r = np.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

            phi = self.np_random.uniform(self.phi_min, self.phi_max)
            x, y = r * np.cos(phi), r * np.sin(phi)

            target = np.array([x, y], dtype=float)
            if np.linalg.norm(target - tip0) >= self.min_tip_dist:
                return [float(x), float(y)]

    # ---------------- IK + nominal control (Residual) ----------------

    def _ik_2link(self, target_xy: np.ndarray, elbow="auto"):
        x, y = float(target_xy[0]), float(target_xy[1])
        r2 = x * x + y * y

        # cos(theta2)
        c2 = (r2 - self.l1 * self.l1 - self.l2 * self.l2) / (2.0 * self.l1 * self.l2)
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2_pos = float(np.sqrt(max(0.0, 1.0 - c2 * c2)))
        s2_neg = -s2_pos

        # two solutions
        th2_up = float(np.arctan2(s2_pos, c2))
        th2_down = float(np.arctan2(s2_neg, c2))

        def th1_from(th2):
            k1 = self.l1 + self.l2 * np.cos(th2)
            k2 = self.l2 * np.sin(th2)
            return float(np.arctan2(y, x) - np.arctan2(k2, k1))

        th1_up = th1_from(th2_up)
        th1_down = th1_from(th2_down)

        cand = {
            "up": np.array([th1_up, th2_up], dtype=float),
            "down": np.array([th1_down, th2_down], dtype=float),
        }

        if elbow in ("up", "down"):
            q = cand[elbow]
        else:
            # auto: pick closest to current q (avoid jumps)
            q_cur = np.array([self.theta1, self.theta2], dtype=float)
            du = np.linalg.norm(
                [
                    _wrap_to_pi(cand["up"][0] - q_cur[0]),
                    _wrap_to_pi(cand["up"][1] - q_cur[1]),
                ]
            )
            dd = np.linalg.norm(
                [
                    _wrap_to_pi(cand["down"][0] - q_cur[0]),
                    _wrap_to_pi(cand["down"][1] - q_cur[1]),
                ]
            )
            q = cand["up"] if du <= dd else cand["down"]

        # clip to [-pi, pi]
        q[0] = float(np.clip(q[0], -np.pi, np.pi))
        q[1] = float(np.clip(q[1], -np.pi, np.pi))
        return q

    def _nominal_pd_torque(self, q_des: np.ndarray):
        # PD em juntas para rastrear q_des (qdot_des = 0)
        q = np.array([self.theta1, self.theta2], dtype=float)
        qdot = np.array([self.qdot1, self.qdot2], dtype=float)

        e = np.array(
            [_wrap_to_pi(q_des[0] - q[0]), _wrap_to_pi(q_des[1] - q[1])], dtype=float
        )
        tau = self.kp * e - self.kd * qdot
        return tau

    # ---------------- Safety filter (proposta completa) ----------------

    def _safety_filter(self, tau_raw: np.ndarray):
        tau_raw = np.asarray(tau_raw, dtype=float).reshape(
            2,
        )
        tau0 = tau_raw.copy()

        tau_out = tau_raw.copy()

        # 1) projeção no box |tau|<=tau_max e rate box |tau - prev|<=dtau_max
        if self.safety_filter in ("proj_box", "proj_box_jointlimit"):
            # clip magnitude
            tau_out = np.clip(tau_out, -self.tau_max, self.tau_max)

            # clip rate
            if self.dtau_max is not None and self.dtau_max > 0:
                lo = self.prev_tau_applied - self.dtau_max
                hi = self.prev_tau_applied + self.dtau_max
                tau_out = np.clip(tau_out, lo, hi)

            # clip again magnitude
            tau_out = np.clip(tau_out, -self.tau_max, self.tau_max)

        # 2) joint-limit “do not push further” near limits
        if self.safety_filter == "proj_box_jointlimit":
            q = np.array([self.theta1, self.theta2], dtype=float)
            qlim = np.pi

            for i in (0, 1):
                if q[i] >= (qlim - self.q_margin) and tau_out[i] > 0:
                    tau_out[i] = 0.0
                if q[i] <= (-qlim + self.q_margin) and tau_out[i] < 0:
                    tau_out[i] = 0.0

        # intervention metrics
        delta = tau_out - tau0
        intervened = int(np.linalg.norm(delta) > 1e-9)

        # update memory (what we *applied*)
        self.prev_tau_applied = tau_out.copy()

        return tau_out, {
            "filter_intervened": intervened,
            "filter_delta_tau_norm": float(np.linalg.norm(delta)),
            "filter_delta_tau1": float(delta[0]),
            "filter_delta_tau2": float(delta[1]),
        }
