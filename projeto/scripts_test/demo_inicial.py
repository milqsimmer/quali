"""
Script de demonstração inicial do braço robótico 2-DOF.

Usa PyBullet e uma solução analítica de cinemática inversa para testar o URDF

e a simulação, sem envolver o ambiente Gym nem o pipeline de aprendizado por
reforço (PPO).

Mantido como ferramenta de debug visual rápido do modelo físico e do setup
do PyBullet.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time


def compute_angles(x, y, l1=0.5, l2=0.5):
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if np.abs(D) > 1.0:
        print("Alvo fora do alcance")
        return None

    theta2 = np.arccos(D)  # cotovelo para baixo (padrão)

    # Agora θ1:
    phi = np.arctan2(y, x)
    psi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = phi - psi

    return theta1, theta2


# Inicia a simulação
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# Carrega o braço com dois links
robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

# Cria o alvo (bolinha vermelha)
target_pos = [0.6, 0.2, 0.1]
target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseVisualShapeIndex=target_vis, basePosition=target_pos)

# Índices das juntas e do link final
joint1 = 0
joint2 = 1
link2 = 1  # o segundo link (cotovelo → ponta)

step = 0
max_steps = 200
success_threshold = 0.01

while step < max_steps:
    # Ângulos oscilando
    # angle1 = np.sin(step * 0.01) * 1.5 # trajetoria oscilando mas nao tenta chegar no target, ainda nao tem objetivo
    # angle2 = np.cos(step * 0.01) * 1.5

    target_x = 0.6
    target_y = 0.2

    angles = compute_angles(target_x, target_y)
    if angles is None:
        print("Alvo fora do alcance para a posição alvo atual; pulando este passo.")
        step += 1
        continue

    theta1, theta2 = angles

    # Controla as juntas
    p.setJointMotorControl2(
        robot, joint1, p.POSITION_CONTROL, targetPosition=theta1, force=20
    )
    p.setJointMotorControl2(
        robot, joint2, p.POSITION_CONTROL, targetPosition=theta2, force=20
    )

    p.stepSimulation()
    time.sleep(1 / 240)

    # Pega a posição da ponta do braço (offset da base do link2)
    link_state = p.getLinkState(robot, link2, computeForwardKinematics=True)
    link_pos = link_state[0]
    link_orient = link_state[1]

    # Corrige a posição da ponta do braço: offset de 0.5m no eixo X local
    offset_local = [0.5, 0, 0]
    end_effector_pos, _ = p.multiplyTransforms(
        link_pos, link_orient, offset_local, [0, 0, 0, 1]
    )

    # Calcula distância ao alvo
    dist = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))
    reward = -dist

    if dist < success_threshold:
        print(
            f"Alvo alcançado com dist = {dist:.4f} em {step} steps; encerrando simulação."
        )
        break

    # Log
    js1 = p.getJointState(robot, joint1)
    js2 = p.getJointState(robot, joint2)
    angle1 = js1[0]
    angle2 = js2[0]

    print(f"Step {step}")
    print(f"  Ângulo1: {angle1:.2f}, Ângulo2: {angle2:.2f}")
    print(f"  Posição da ponta: {np.round(end_effector_pos, 3)}")
    print(f"  Distância até alvo: {dist:.4f}")
    print(f"  Recompensa: {reward:.4f}")
    print("-" * 40)

    step += 1
