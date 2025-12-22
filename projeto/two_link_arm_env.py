# two_link_arm_env.py (TORQUE_CONTROL only)

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class TwoLinkArmEnv(gym.Env):
    """
    Two-link planar arm in PyBullet with direct torque control.

    Action: tau = [tau1, tau2] (N·m), clipped to [-tau_max, tau_max]
    Observation: [theta1, theta2, target_x, target_y]
    """

    def __init__(
        self,
        render: bool = False,
        reward_mode: str = "pure",  # "pure" ou "pirl"
        seed: int | None = None,
        # ====== alvo ======
        margin: float = 0.02,
        min_tip_dist: float = 0.07,
        phi_min: float = -np.pi / 2,
        phi_max: float = np.pi / 2,
        # ====== torque control ======
        tau_max: float = 20.0,
        lam_a: float = 0.001,  # penalidade de ação (||tau||)
        alpha_tau: float = 0.0005,  # peso do termo físico no PIRL (||tau_applied||_1)
    ):
        super().__init__()

        assert reward_mode in ("pure", "pirl"), "reward_mode deve ser 'pure' ou 'pirl'"

        self.reward_mode = reward_mode
        self.render_mode = render

        # parâmetros do alvo
        self.margin = float(margin)
        self.min_tip_dist = float(min_tip_dist)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        # torque control
        self.tau_max = float(tau_max)
        self.lam_a = float(lam_a)
        self.alpha_tau = float(alpha_tau)

        # PyBullet init
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

        self.joint_ids = [0, 1]
        self.link2 = 1

        # comprimentos dos elos
        self.l1 = 0.5
        self.l2 = 0.5
        self.offset_local = [self.l2, 0, 0]

        # Obs: [theta1, theta2, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -1.5, -1.5], dtype=np.float32),
            high=np.array([np.pi, np.pi, 1.5, 1.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Ação: torques
        self.action_space = spaces.Box(
            low=np.array([-self.tau_max, -self.tau_max], dtype=np.float32),
            high=np.array([self.tau_max, self.tau_max], dtype=np.float32),
            dtype=np.float32,
        )

        # RNG
        self.np_random, _ = seeding.np_random(seed)

        # estado
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.target_pos = None
        self.d0 = None

        # desabilita motores padrão para permitir torque direto
        self._disable_motors()

    # ---------------- Gym API ----------------

    def reset(self, *, seed=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        # reseta juntas (pos e vel)
        for j in self.joint_ids:
            p.resetJointState(self.robot, j, targetValue=0.0, targetVelocity=0.0)

        # garante torque control (motores desabilitados)
        self._disable_motors()

        # aplica torque zero no início
        self._apply_torque(np.array([0.0, 0.0], dtype=np.float32))
        p.stepSimulation()

        # estado real
        js = p.getJointStates(self.robot, self.joint_ids)
        self.theta1 = float(js[0][0])
        self.theta2 = float(js[1][0])

        # tip0 real após reset
        tip0 = np.array(self._get_end_effector_pos()[:2], dtype=float)

        # amostra alvo e calcula dificuldade
        self.target_pos = self._sample_target(tip0=tip0)
        self.d0 = float(np.linalg.norm(tip0 - np.array(self.target_pos, dtype=float)))

        self._create_target_visual()
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            2,
        )
        tau_cmd = np.clip(action, -self.tau_max, self.tau_max)

        # aplica torque
        self._apply_torque(tau_cmd)
        p.stepSimulation()

        # lê estado real e torques aplicados reportados
        js = p.getJointStates(self.robot, self.joint_ids)
        self.theta1 = float(js[0][0])
        self.theta2 = float(js[1][0])

        # torque "aplicado" (reportado) — pode diferir do comandado dependendo do solver
        tau1_applied = float(abs(js[0][3]))
        tau2_applied = float(abs(js[1][3]))
        tau_l1 = float(tau1_applied + tau2_applied)

        # distância ponta–alvo
        ee = self._get_end_effector_pos()
        dist = float(
            np.linalg.norm(
                np.array(ee[:2], dtype=float) - np.array(self.target_pos, dtype=float)
            )
        )

        # ====== REWARD (decomposto) ======
        r_dist = -dist
        r_act = -self.lam_a * float(
            np.linalg.norm(tau_cmd)
        )  # penaliza magnitude do torque comandado

        if self.reward_mode == "pure":
            r_tau = 0.0
            reward = r_dist + r_act
            alpha_tau_used = 0.0
        else:
            # termo físico: penaliza esforço (torque reportado) - proxy
            r_tau = -self.alpha_tau * tau_l1
            reward = r_dist + r_act + r_tau
            alpha_tau_used = self.alpha_tau

        done = dist < 0.05

        info = {
            "distance": dist,
            "d0": float(self.d0) if self.d0 is not None else np.nan,
            # torques
            "tau1": tau1_applied,
            "tau2": tau2_applied,
            "tau_l1": tau_l1,
            # torque comandado (útil p/ auditoria)
            "tau_cmd1": float(tau_cmd[0]),
            "tau_cmd2": float(tau_cmd[1]),
            "tau_cmd_l1": float(abs(tau_cmd[0]) + abs(tau_cmd[1])),
            # componentes do reward
            "r_dist": float(r_dist),
            "r_act": float(r_act),
            "r_tau": float(r_tau),
            "lam_a": float(self.lam_a),
            "alpha_tau": float(alpha_tau_used),
        }

        return self._get_obs(), float(reward), done, info

    def render(self, mode="human"):
        if self.render_mode:
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect(self.physicsClient)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    # ---------------- Alvo / obs ----------------

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

    # ---------------- PyBullet helpers ----------------

    def _disable_motors(self):
        # desabilita os motores padrão para permitir torque direto
        for j in self.joint_ids:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)

    def _apply_torque(self, tau: np.ndarray):
        tau = np.asarray(tau, dtype=np.float32).reshape(
            2,
        )
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        p.setJointMotorControl2(self.robot, 0, p.TORQUE_CONTROL, force=float(tau[0]))
        p.setJointMotorControl2(self.robot, 1, p.TORQUE_CONTROL, force=float(tau[1]))

    def _get_end_effector_pos(self):
        ls = p.getLinkState(self.robot, self.link2, computeForwardKinematics=True)
        link_pos = ls[4] if len(ls) > 4 else ls[0]
        link_ornt = ls[5] if len(ls) > 5 else ls[1]
        ee_pos, _ = p.multiplyTransforms(
            link_pos, link_ornt, self.offset_local, [0, 0, 0, 1]
        )
        return ee_pos
