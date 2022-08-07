from pathlib import Path
import gym
from gym import ObservationWrapper
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
import matplotlib.pyplot as plt

import time
import line_profiler

from agent import DQNAgent
from utils import USE_CUDA, TransformObservation, ENV_NAME

env_name = ENV_NAME

gamma = 0.99
max_experiences = 10**5
min_experiences = 512
batch_size = 64
#epsilon = 1e-7
#lr = 1e-4

num_episodes = 1000

class Trainer:
    def __init__(self,env,agent):
        self.env = env
        self.agent = agent
        self.reward_history = []
        self.loss_history = []

    def train(self,num_episodes,verbose=1):
        env = self.env
        agent = self.agent
        for n in range(1,num_episodes+1):
            state = env.reset()
            epsilon = agent.get_epsilon(n)
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = agent.get_action(state,epsilon)
                next_state,reward,done,info = env.step(action)
                total_reward += reward

                agent.add_experience((state,action,reward,next_state,done))
                loss = agent.update_qnetwork()
                self.loss_history.append(loss)
                state = next_state
                step += 1
            agent.target_update()
            self.reward_history.append(total_reward)
            if verbose == 1:
                print(f"Episodes {n}:{total_reward}")

        if verbose == 1:
            agent.save()
            print("train is finished.")

    def reward_plot(self,average=1):
        path = Path(Path.cwd(),Path("logs"))
        if not path.exists():
            path.mkdir()
        fig = plt.figure()
        plt.plot(range(1,len(self.reward_history)+1),self.reward_history,alpha=0.4,c="orange")
        plt.xlabel("Episodes")
        plt.ylabel("Total reward")

        average_history = []
        size = len(self.reward_history)
        tmp = 0
        count = 0
        for i,a in enumerate(self.reward_history):
            tmp += a
            count += 1
            if (i+1)%average==0 or (i+1)==size:
                average_history.append(tmp/count)
                tmp=0
                count=0
        x = np.arange(1,size+1,average)
        if x[-1]!=size:
            np.append(x,size)
        plt.plot(x,average_history)
        plt.savefig("./logs/reward_log")

def main():
    env = gym.make(env_name)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = TransformObservation(env)
    agent = DQNAgent(env=env,
                     gamma=gamma,
                     max_experiences=max_experiences,
                     min_experiences=min_experiences,
                     batch_size=batch_size,
                    )

    trainer = Trainer(env,agent)
    trainer.train(num_episodes)
    trainer.reward_plot(average=10)

if __name__ == '__main__':
    """
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time:{elapsed_time} sec")

    """
    pr = line_profiler.LineProfiler()
    pr.add_function(main)
    pr.add_function(Trainer.train)
    pr.runcall(main)
    print(" ### main program is finished. ###")
    pr.print_stats()
    #"""
