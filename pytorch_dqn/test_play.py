import gym
from gym import wrappers, ObservationWrapper
import gym_minigrid
from gym_minigrid.wrappers import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import time

from agent import DQNAgent
from utils import USE_CUDA, TransformObservation, ENV_NAME

env_name = ENV_NAME
num_tests = 3

def main():
    env = gym.make(env_name)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = TransformObservation(env)
    agent = DQNAgent(env=env)
    agent.load()

    path = Path(Path.cwd(),Path("movies"))
    env = wrappers.Monitor(env,path,force=True,video_callable=(lambda e:True))
    for i in range(num_tests):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state,0)
            next_state,reward,done,info = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Test: {i}")
        print(f"Total_reward: {total_reward}")
        print()

if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time:{elapsed_time} sec")
