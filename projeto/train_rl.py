# train_rl.py  (use este para ambos, mudando reward_mode via CLI)
import argparse, os
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from two_link_arm_env import TwoLinkArmEnv

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["pure", "pirl"], default="pure")
parser.add_argument("--steps", type=int, default=300_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = TwoLinkArmEnv(render=False, reward_mode=args.mode, seed=args.seed)
env = TimeLimit(env, max_episode_steps=200)  # garante corte temporal
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    seed=args.seed,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)
model.learn(total_timesteps=args.steps)

os.makedirs(f"runs_{args.mode}", exist_ok=True)
model.save(f"runs_{args.mode}/ppo_model_seed{args.seed}.zip")
env.close()
print(f"[OK] Treino {args.mode} finalizado.")
