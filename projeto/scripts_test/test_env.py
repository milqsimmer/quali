"""
Teste de ambiente e cinemática inversa com o TwoLinkArmEnv.

Para vários alvos aleatórios gerados pelo próprio ambiente Gym,
resolve a cinemática inversa analítica (theta1, theta2), aplica
os ângulos diretamente no robô via TwoLinkArmEnv e mede a
distância final entre a ponta do braço e o alvo.

Não envolve treino nem política de RL; serve para verificar
se IK, URDF e offset da ponta estão coerentes dentro do env.
"""

from two_link_arm_env import TwoLinkArmEnv
import numpy as np
import time
import pybullet as p


def compute_angles(x, y, l1=0.5, l2=0.5):
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if np.abs(D) > 1.0:
        return None  # fora do alcance

    theta2 = np.arccos(D)
    phi = np.arctan2(y, x)
    psi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = phi - psi

    return theta1, theta2


# Cria o ambiente com render ligado
env = TwoLinkArmEnv(render=True)

num_trials = 10

for i in range(num_trials):
    print(f"\n🎯 Testando alvo {i+1}/{num_trials}")
    obs = env.reset()
    target_x, target_y = obs[2], obs[3]

    angles = compute_angles(target_x, target_y)
    if angles is None:
        print("❌ Alvo fora do alcance")
        continue

    theta1, theta2 = angles
    print(
        f"✅ Solução encontrada: theta1={np.degrees(theta1):.2f}°, theta2={np.degrees(theta2):.2f}°"
    )

    # Aplica os ângulos no robô
    env.theta1 = theta1
    env.theta2 = theta2
    env._apply_angles(theta1, theta2)
    # Dá tempo para o braço alcançar a posição
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1 / 240)  # só se estiver com render ligado

    # Verifica a posição real da ponta
    ee_pos = env._get_end_effector_pos()
    dist = np.linalg.norm(np.array(ee_pos[:2]) - np.array([target_x, target_y]))
    print(f"📍 Distância da ponta ao alvo: {dist:.4f}")

    time.sleep(1)

env.close()
