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


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    suffix = f"_" + run_tag if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}")


def get_model_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), f"ppo_model_{seed}.zip")


def get_monitor_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), "monitor.csv")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["pure", "pirl"], default="pure")
parser.add_argument("--steps", type=int, default=300_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--run-tag",
    type=str,
    default="",
    help="Sufixo opcional para distinguir execucoes com mesma seed.",
)
parser.add_argument(
    "--lambda-a",
    type=float,
    default=0.001,
    dest="lambda_a",
    help="Peso da penalidade de acao na recompensa.",
)
parser.add_argument(
    "--alpha-tau",
    type=float,
    default=0.0005,
    help="Peso da penalidade de torque na recompensa (modo pirl).",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2e-4,
    dest="learning_rate",
    help="Taxa de aprendizado do otimizador PPO (default: 2e-4).",
)
parser.add_argument(
    "--n-steps",
    type=int,
    default=2048,
    dest="n_steps",
    help="Numero de steps por rollout antes de cada atualizacao PPO.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    dest="batch_size",
    help="Tamanho do minibatch usado nas atualizacoes PPO.",
)
parser.add_argument(
    "--n-epochs",
    type=int,
    default=20,
    dest="n_epochs",
    help="Numero de epocas de otimizacao por rollout.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Fator de desconto para retornos (gamma).",
)
parser.add_argument(
    "--gae-lambda",
    type=float,
    default=0.95,
    dest="gae_lambda",
    help="Parametro lambda do GAE (Generalized Advantage Estimation).",
)
parser.add_argument(
    "--clip-range",
    type=float,
    default=0.2,
    dest="clip_range",
    help="Intervalo de clipping da perda PPO.",
)
args = parser.parse_args()

set_global_seed(args.seed)

run_dir = get_run_dir(args.mode, args.seed, args.run_tag)
os.makedirs(run_dir, exist_ok=True)

monitor_path = get_monitor_path(args.mode, args.seed, args.run_tag)
model_path = get_model_path(args.mode, args.seed, args.run_tag)

env = TwoLinkArmEnv(
    render=False,
    reward_mode=args.mode,
    lambda_a=args.lambda_a,
    alpha_tau=args.alpha_tau,
)
env = TimeLimit(env, max_episode_steps=200)

# IMPORTANTE:
# info_keywords registra colunas extras no monitor.csv no fim de cada episódio
env = Monitor(
    env,
    filename=monitor_path,
    info_keywords=(
        "is_success",
        "final_distance",
        "tau_sum_total",
        "episode_energy",
        "episode_mean_tau_sum",
    ),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
)

model.learn(total_timesteps=args.steps)

model.save(model_path)
env.close()

print(f"[OK] Treino {args.mode} finalizado.")
print(f"[OK] Monitor salvo em: {monitor_path}")
print(f"[OK] Modelo salvo em: {model_path}")
