import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class TwoLinkArmEnv(gym.Env):
    def __init__(
        self,
        render: bool = False,
        reward_mode: str = "pure",
        lambda_a: float = 0.001,
        alpha_tau: float = 0.0005,
    ):
        super(TwoLinkArmEnv, self).__init__()
        self.reward_mode = reward_mode

        # pesos da função de recompensa
        self.lambda_a = float(lambda_a)
        self.alpha_tau = float(alpha_tau)

        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # passo de tempo usado pelo motor fisico (para potencia/energia)
        params = p.getPhysicsEngineParameters()
        self.dt = float(params.get("fixedTimeStep", 1.0 / 240.0))

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

        self.link1 = 0
        self.link2 = 1

        self.l1 = 0.5
        self.l2 = 0.5

        self.offset_local = [self.l2, 0, 0]

        self.success_threshold = 0.05

        # Observação: [theta1, theta2, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -1.5, -1.5]),
            high=np.array([np.pi, np.pi, 1.5, 1.5]),
            dtype=np.float32,
        )

        # Ações: variação nos ângulos das juntas
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1]),
            dtype=np.float32,
        )

        self.target_pos = None
        self.state = None
        self.target_id = None

        # acumuladores por episódio (para métricas agregadas)
        self.ep_tau_sum_total = 0.0
        self.ep_energy = 0.0
        self.ep_steps = 0

        self.reset()

    def reset(self):
        self.theta1 = 0.0
        self.theta2 = 0.0

        self.target_pos = self._sample_target()

         # zera acumuladores por episódio
        self.ep_tau_sum_total = 0.0
        self.ep_energy = 0.0
        self.ep_steps = 0

        # coloca as juntas exatamente em (theta1, theta2) e zera velocidades
        p.resetJointState(self.robot, 0, self.theta1, targetVelocity=0.0)
        p.resetJointState(self.robot, 1, self.theta2, targetVelocity=0.0)

        # configura os motores de posição para manter esses ângulos
        self._apply_angles(self.theta1, self.theta2)

        self._create_target_visual()

        obs = self._get_obs()
        return obs

    def step(self, action):
        # interpreta a ação como incremento em ângulo sobre o estado ATUAL
        # (estado atual é sempre sincronizado com o Bullet via getJointStates)
        th1_target = np.clip(self.theta1 + action[0], -np.pi, np.pi)
        th2_target = np.clip(self.theta2 + action[1], -np.pi, np.pi)

        # aplica ângulos alvo e avança simulação física
        self._apply_angles(th1_target, th2_target)
        p.stepSimulation()

        # lê estado REAL das juntas após o passo
        js = p.getJointStates(self.robot, [0, 1])
        joint1_pos, joint1_vel, _, joint1_tau = js[0]
        joint2_pos, joint2_vel, _, joint2_tau = js[1]

        # sincroniza estado interno com o Bullet
        self.theta1 = float(joint1_pos)
        self.theta2 = float(joint2_pos)

        # distância ponta–alvo no plano (x, y)
        end_effector_pos = self._get_end_effector_pos()
        dist = np.linalg.norm(
            np.array(end_effector_pos[:2]) - np.array(self.target_pos)
        )

        # torques e velocidades físicos
        tau1 = abs(float(joint1_tau))
        tau2 = abs(float(joint2_tau))
        tau_sum = tau1 + tau2  # soma de |tau1| + |tau2|

        vel1 = float(joint1_vel)
        vel2 = float(joint2_vel)
        power = abs(tau1 * vel1) + abs(tau2 * vel2)

        # atualiza acumuladores por episódio
        if not np.isnan(tau_sum):
            self.ep_tau_sum_total += tau_sum
        if not np.isnan(power):
            self.ep_energy += power * self.dt
        self.ep_steps += 1

        # ====== REWARD ======
        # RL puro: -distância + penalidade leve de ação
        lam_a = self.lambda_a

        if self.reward_mode == "pure":
            reward = -dist - lam_a * float(np.linalg.norm(action))
        else:  # "pirl"
            alpha_tau = self.alpha_tau
            reward = -dist - lam_a * float(np.linalg.norm(action)) - alpha_tau * tau_sum

        done = dist < self.success_threshold

        mean_tau = (
            self.ep_tau_sum_total / self.ep_steps if self.ep_steps > 0 else float("nan")
        )

        obs = self._get_obs()
        info = {
            "distance": float(dist),
            "final_distance": float(dist),
            "is_success": float(dist < self.success_threshold),
            "tau1": tau1,
            "tau2": tau2,
            "tau_sum": tau_sum,
            "omega1": vel1,
            "omega2": vel2,
            "power": power,
            # métricas acumuladas por episódio
            "tau_sum_total": float(self.ep_tau_sum_total),
            "episode_energy": float(self.ep_energy),
            "episode_mean_tau_sum": float(mean_tau),
        }

        return obs, reward, done, info

    def _create_target_visual(self):
        if self.target_id is not None:
            p.removeBody(self.target_id)

        radius = 0.03
        red = [1, 0, 0, 1]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=red)
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=vis,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.1],
        )

    def _sample_target(self):
        while True:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(-0.5, 0.5)
            if np.hypot(x, y) <= (self.l1 + self.l2):
                return [x, y]

    def _apply_angles(self, theta1, theta2):
        p.setJointMotorControl2(
            self.robot, 0, p.POSITION_CONTROL, targetPosition=theta1, force=20
        )
        p.setJointMotorControl2(
            self.robot, 1, p.POSITION_CONTROL, targetPosition=theta2, force=20
        )

    def _get_end_effector_pos(self):
        ls = p.getLinkState(self.robot, self.link2, computeForwardKinematics=True)
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
