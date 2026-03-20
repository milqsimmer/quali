"""
Sanity check rápido de dependências principais.
- Imprime as versões de Gym e NumPy.
- Cria o ambiente CartPole-v1 e executa alguns passos aleatórios.
Serve apenas para verificar se Gym e NumPy estão instalados e
funcionando antes de rodar os scripts do braço robótico e de RL.
"""

import gym

print("Gym:", gym.__version__)
import numpy as np

print("NumPy:", np.__version__)
env = gym.make("CartPole-v1")
obs = env.reset()
for _ in range(5):
    obs, r, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
print("OK")
