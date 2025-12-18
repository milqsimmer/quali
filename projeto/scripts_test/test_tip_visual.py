# test_tip_visual.py  (gym 0.21)
import time, numpy as np, pybullet as p
from two_link_arm_env import TwoLinkArmEnv

EPISODES = 1
STEPS = 300

env = TwoLinkArmEnv(render=True)  # render=True para ver a GUI
obs = env.reset()

text_id = None
line_id = None

for step in range(1, STEPS + 1):
    # ação pequena só pra "passear"
    a = env.action_space.sample() * 0.1
    obs, r, done, info = env.step(a)

    # pega ponta (já deve ser a ponta no env TIP)
    ee = np.array(env._get_end_effector_pos()[:2], dtype=float)
    tgt = np.array(env.target_pos[:2], dtype=float)
    dist = float(np.linalg.norm(ee - tgt))

    if step % 10 == 0 or done:
        print(
            f"step={step:03d} | tip_xy={ee.round(4)} | tgt_xy={tgt.round(4)} | dist_tip={dist:.4f}"
        )

    # overlay: texto da distância e linha ponta->alvo
    try:
        if text_id is not None:
            p.removeUserDebugItem(text_id)
        if line_id is not None:
            p.removeUserDebugItem(line_id)
        text_id = p.addUserDebugText(
            f"dist_tip = {dist:.4f} m",
            textPosition=[float(tgt[0]), float(tgt[1]), 0.25],
            textColorRGB=[0, 0, 0],
            textSize=1.4,
            lifeTime=0,
        )
        line_id = p.addUserDebugLine(
            [float(ee[0]), float(ee[1]), 0.1],
            [float(tgt[0]), float(tgt[1]), 0.1],
            lineColorRGB=[0, 0, 1],
            lineWidth=2,
            lifeTime=0,
        )
    except Exception:
        pass

    time.sleep(1 / 240.0)
    if done:
        obs = env.reset()

env.close()
