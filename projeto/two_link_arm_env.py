# two_link_arm_env.py

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class TwoLinkArmEnv(gym.Env):
    def __init__(
        self,
        render=False,
        reward_mode="pure",
        seed: int | None = None,
        margin: float = 0.02,
        min_tip_dist: float = 0.07,
        phi_min: float = -np.pi / 2,
        phi_max: float = np.pi / 2,
    ):
        super(TwoLinkArmEnv, self).__init__()
        self.reward_mode = reward_mode

        # ---- parâmetros do sampler (A2) ----
        self.margin = float(margin)
        self.min_tip_dist = float(min_tip_dist)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        # ------------------------------------

        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

        self.link1 = 0
        self.link2 = 1

        self.l1 = 0.5
        self.l2 = 0.5

        self.offset_local = [self.l2, 0, 0]

        # Observação: [theta1, theta2, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -1.5, -1.5]),
            high=np.array([np.pi, np.pi, 1.5, 1.5]),
            dtype=np.float32,
        )

        # Ações: variação nos ângulos das juntas
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1]), high=np.array([0.1, 0.1]), dtype=np.float32
        )

        self.np_random, _ = seeding.np_random(seed)

        self.target_pos = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.theta1 = 0.0
        self.theta2 = 0.0

        # zera estado físico das juntas
        p.resetJointState(self.robot, 0, targetValue=0.0, targetVelocity=0.0)
        p.resetJointState(self.robot, 1, targetValue=0.0, targetVelocity=0.0)

        # self._apply_angles(self.theta1, self.theta2)
        p.stepSimulation()

        # posição inicial da ponta do braço
        tip0 = np.array(self._get_end_effector_pos()[:2])

        self.target_pos = self._sample_target(tip0)
        self.d0 = float(np.linalg.norm(tip0 - np.array(self.target_pos)))

        self._create_target_visual()  # Adiciona a bolinha vermelha (alvo)

        obs = self._get_obs()
        return obs

    def step(self, action):
        # atualiza ângulos com incremento da ação
        self.theta1 = np.clip(self.theta1 + action[0], -np.pi, np.pi)
        self.theta2 = np.clip(self.theta2 + action[1], -np.pi, np.pi)

        # aplica ângulos e avança simulação
        self._apply_angles(self.theta1, self.theta2)
        p.stepSimulation()

        # distância ponta–alvo no plano (x, y)
        end_effector_pos = self._get_end_effector_pos()
        dist = np.linalg.norm(
            np.array(end_effector_pos[:2]) - np.array(self.target_pos)
        )

        # lê torques aplicados nas duas juntas
        js = p.getJointStates(self.robot, [0, 1])
        # appliedMotorTorque é o quarto elemento do tuple
        tau1 = abs(js[0][3])
        tau2 = abs(js[1][3])
        tau_l1 = tau1 + tau2  # ||tau_t||_1 = |tau1| + |tau2|

        # ====== REWARD (decomposto) ======
        lam_a = 0.001

        # componentes comuns
        r_dist = -float(dist)
        r_act = -lam_a * float(np.linalg.norm(action))

        if self.reward_mode == "pure":
            alpha_tau = 0.0
            r_tau = 0.0
            reward = r_dist + r_act
        else:  # "pirl"
            alpha_tau = 0.0005  # ajuste fino depois
            r_tau = -alpha_tau * float(tau_l1)
            reward = r_dist + r_act + r_tau

        done = dist < 0.05

        obs = self._get_obs()
        info = {
            "distance": float(dist),
            "d0": float(getattr(self, "d0", np.nan)),
            "tau1": float(tau1),
            "tau2": float(tau2),
            "tau_l1": float(tau_l1),
            # --- decomposição do reward ---
            "r_dist": float(r_dist),
            "r_act": float(r_act),
            "r_tau": float(r_tau),
            "lam_a": float(lam_a),
            "alpha_tau": float(alpha_tau),
        }
        return obs, float(reward), done, info

    def _create_target_visual(self):
        # Apaga alvo anterior (se houver)
        if hasattr(self, "target_id"):
            p.removeBody(self.target_id)

        radius = 0.03
        red = [1, 0, 0, 1]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=red)
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=vis,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.1],
        )

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _sample_target(self, tip0, margin=0.02, min_tip_dist=0.07):
        r_min = abs(self.l1 - self.l2) + margin
        r_max = (self.l1 + self.l2) - margin

        while True:
            u = self.np_random.uniform(0.0, 1.0)
            r = np.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

            phi = self.np_random.uniform(-np.pi / 2, np.pi / 2)  # x > 0
            x, y = r * np.cos(phi), r * np.sin(phi)

            target = np.array([x, y])

            if np.linalg.norm(target - tip0) >= min_tip_dist:
                return [float(x), float(y)]

    def _apply_angles(self, theta1, theta2):
        p.setJointMotorControl2(
            self.robot, 0, p.POSITION_CONTROL, targetPosition=theta1, force=20
        )
        p.setJointMotorControl2(
            self.robot, 1, p.POSITION_CONTROL, targetPosition=theta2, force=20
        )

    def _get_end_effector_pos(self):
        ls = p.getLinkState(self.robot, self.link2, computeForwardKinematics=True)
        # use 4/5 se disponíveis; caso contrário, caia em 0/1
        link_pos = ls[4] if len(ls) > 4 else ls[0]
        link_ornt = ls[5] if len(ls) > 5 else ls[1]
        ee_pos, _ = p.multiplyTransforms(
            link_pos, link_ornt, self.offset_local, [0, 0, 0, 1]
        )
        return ee_pos

    def _get_obs(self):
        return np.array(
            [self.theta1, self.theta2, self.target_pos[0], self.target_pos[1]],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        if self.render_mode:
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect(self.physicsClient)
