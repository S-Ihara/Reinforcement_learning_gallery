from pathlib import Path
import numpy as np
import torch

from model import CNNQNet
from buffer import SimpleReplayBuffer
from utils import Variable, USE_CUDA, DEVICE

class DQNAgent:
    def __init__(self,env,gamma=0.99,max_experiences=10000,min_experiences=500,batch_size=64):
        """
        Arguments:
            env: gym.Env
            ハイパラ
            gamma: float 割引率
            max_experiences: int リプレイバッファの最大経験備蓄数
            min_experiences: int 学習初めに必要な経験数
            batch_size: int 学習のバッチサイズ

            # TODO: optim周りのハイパラ

        """
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.env = env
        self.input_features = self.env.observation_space.shape # TODO: envに合わせて変える必要あるのをなんとかする
        #self.num_actions = self.env.action_space.n
        self.num_actions = 3 # 今回は学習の簡易化のために行動次元数を減らす（いらん行動を消す）
        self.gamma = gamma
        self.Q = CNNQNet(self.input_features,self.num_actions)
        self.target_Q = CNNQNet(self.input_features,self.num_actions)
        self.Q.to(DEVICE)
        self.target_Q.to(DEVICE)
        self.replay_buffer = SimpleReplayBuffer(state_shape=self.env.observation_space.shape,
                                                action_shape=1,
                                                size=self.max_experiences)
        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr=1e-4,eps=1e-7)

    def get_epsilon(self,num_episode):
        return max(0.05,0.5-num_episode*0.02)

    def get_action(self,state,epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state,dtype=torch.float).div_(255).unsqueeze(0)
                action = torch.argmax(self.Q(state)).item()
        return action

    def update_qnetwork(self):
        if len(self.replay_buffer) < self.min_experiences:
            return
        (states,actions,rewards,next_states,dones) = self.replay_buffer.get_minibatch(self.batch_size)

        states      = Variable(torch.from_numpy(states)).div_(255)
        actions     = Variable(torch.tensor(actions))
        rewards     = Variable(torch.tensor(rewards))
        next_states = Variable(torch.from_numpy(next_states)).div_(255)
        dones       = Variable(torch.tensor(dones, dtype=torch.bool))

        if USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()

        current_Q_values = self.Q(states).gather(1,actions.type(torch.int64))
        next_max_q = self.target_Q(next_states).detach().max(1)[0].unsqueeze(1)
        next_Q_values = ~dones * next_max_q
        target_Q_values = rewards + (self.gamma * next_Q_values)

        loss = (target_Q_values - current_Q_values) ** 2
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_experience(self,exp):
        self.replay_buffer.push(exp)

    def target_update(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def save(self):
        pathdir = Path(Path.cwd(),Path("models"))
        if not pathdir.exists():
            pathdir.mkdir()
        path = Path(pathdir,Path("q.pth"))
        torch.save(self.Q.state_dict(),path)

    def load(self):
        path = Path(Path.cwd(),Path("models/q.pth"))
        self.Q.load_state_dict(torch.load(path))
        #self.Q.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
