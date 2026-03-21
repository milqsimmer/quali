import argparse
import os
import random
import numpy as np
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from two_link_arm_env import TwoLinkArmEnv


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_run_dir(mode: str, seed: int) -> str:
    return os.path.join(f"runs_{mode}", f"seed_{seed}")


def get_model_path(mode: str, seed: int) -> str:
    return os.path.join(get_run_dir(mode, seed), f"ppo_model_{seed}.zip")


def get_monitor_path(mode: str, seed: int) -> str:
    return os.path.join(get_run_dir(mode, seed), "monitor.csv")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["pure", "pirl"], default="pure")
parser.add_argument("--steps", type=int, default=300_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

set_global_seed(args.seed)

run_dir = get_run_dir(args.mode, args.seed)
os.makedirs(run_dir, exist_ok=True)

monitor_path = get_monitor_path(args.mode, args.seed)
model_path = get_model_path(args.mode, args.seed)

env = TwoLinkArmEnv(render=False, reward_mode=args.mode)
env = TimeLimit(env, max_episode_steps=200)

# IMPORTANTE:
# info_keywords registra colunas extras no monitor.csv no fim de cada episódio
env = Monitor(
    env,
    filename=monitor_path,
    info_keywords=("is_success", "final_distance"),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

model.learn(total_timesteps=args.steps)

model.save(model_path)
env.close()

print(f"[OK] Treino {args.mode} finalizado.")
print(f"[OK] Monitor salvo em: {monitor_path}")
print(f"[OK] Modelo salvo em: {model_path}")
