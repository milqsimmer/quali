# eval_rl.py
import numpy as np, time
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from two_link_arm_env import TwoLinkArmEnv


def evaluate(mode, episodes=50, render=False):
    env = TwoLinkArmEnv(render=render, reward_mode=mode)
    env = TimeLimit(env, max_episode_steps=200)
    model = PPO.load(f"runs_{mode}/ppo_model.zip", env=env)

    succ, dists, rets, lens = 0, [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            if render:
                time.sleep(1 / 240.0)
        dists.append(info.get("distance", np.nan))
        rets.append(ep_ret)
        lens.append(ep_len)
        if info.get("distance", 1e9) < 0.05:
            succ += 1
    env.close()
    return {
        "mode": mode,
        "success_rate": succ / episodes,
        "mean_final_dist": float(np.nanmean(dists)),
        "mean_return": float(np.mean(rets)),
        "mean_ep_len": float(np.mean(lens)),
    }


if __name__ == "__main__":
    for m in ["pure", "pirl"]:
        stats = evaluate(m, episodes=100, render=False)
        print(stats)
