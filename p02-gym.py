import gym
from gym import wrappers
import os
import numpy as np
import torch
from shared.logger import get_start_time
from shared.gym.summary import show_summary_data
from matplotlib import pyplot as plt

PROGRAM = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT = os.path.join("output", "gym")
OUTFILE = os.path.join(OUTPUT, "{}-{}".format(PROGRAM, get_start_time()))
EPISODES = 50
STEPS = 300

print("Running: {}".format(PROGRAM))
print("Saving Video At: {}".format(OUTFILE))

env = gym.make('CartPole-v0')
env.reset()
env = wrappers.Monitor(env, OUTFILE)

durations = np.array([], dtype = int)
rewards = np.array([], dtype = float)
parameters = []

# run an episode
def run_episode(env, params):
    observation = torch.tensor(env.reset(), dtype = torch.double)
    total_reward = 0
    print("Starting Episode {}".format(episode))

    for step in range(STEPS):
        result = torch.sum(torch.mul(observation.double(), params.double()))
        action = 0 if result < 0 else 1

        observation, reward, done, info = env.step(action)
        total_reward += reward

        # game ends
        if done:
            print("Game/Episode Over at step {}".format(step))
            break
    return step, total_reward, params

# loop through episodes capturing results and parameters
for episode in range(EPISODES):
    params = torch.sub(torch.mul(torch.randn(env.observation_space.shape[0]), 2), 1)
    duration, reward, _ = run_episode(env, params)

    durations = np.append(durations, duration)
    rewards = np.append(rewards, reward)

env.close()

show_summary_data(EPISODES, rewards, durations)

plt.hist(duration)
plt.show(block = True)