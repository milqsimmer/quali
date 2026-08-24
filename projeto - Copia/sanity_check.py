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
