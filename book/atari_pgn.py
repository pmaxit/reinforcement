import sys
import time
import numpy as np

import torch
import torch.nn as nn
import gym
import ptan
import numpy as np
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 10
BASELINE_STEPS = 1000000
GRAD_L2_CLIP = 0.1

ENV_COUNT = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AtariPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)




def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


def train(env, agent):
    envs = [make_env() for _ in range(ENV_COUNT)]
    net = agent.model
    print(net)

    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_cout = REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0
    baseline_buf = MeanBuffer(BASELINE_STEPS)

    batch_states, batch_actions, batch_scales = [], [], []
    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    m_grad_max, m_grad_mean = [], []
    sum_reward = 0.0
    total_rewards = []

    for step_idx, exp in enumerate(exp_source):
        baseline_buf.add(exp.reward)
        baseline = baseline_buf.mean()
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()

        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print(" %d: reward : %6.2f , mean: %6.2f, episodes: %d" %(step_idx, reward, mean_rewards, done_episodes))

            if mean_rewards > 195:
                print('Solved in %d steps and %d episodes !'%(step_idx, done_episodes))
                break


        if len(batch_states) < BATCH_SIZE:
            continue

        train_step_idx += 1
        states_v = torch.FloatTensor(np.array(batch_states, copy=False)).to(device)
        batch_actions_t = torch.LongTensor(batch_actions).to(device)

        scale_std = np.std(batch_scales)
        batch_scale_v = torch.FloatTensor(batch_scales).to(device)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()
        
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
        optimizer.step()

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()


if __name__== '__main__':
    env = [make_env() for _ in range(ENV_COUNT)]
    net = AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)

    train(env, agent)