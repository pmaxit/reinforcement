import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from gym.wrappers import Monitor

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 16

REWARD_STEPS = 10

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def train(env, agent):

    net = agent.model

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma = GAMMA, steps_count = REWARD_STEPS
    )

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions , batch_scales =[], [], []

    for step_idx, exp in enumerate(exp_source):

        reward_sum += exp.reward
        baseline   =  reward_sum / (step_idx + 1)
        batch_states.append(exp.state)
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

        if len(batch_actions) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

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
        optimizer.step()

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    return net

def test(env, agent):
    # test the agent
    N_GAMES = 100
    for i in range(N_GAMES):
        state  = env.reset()
        done = False
        rewards = []
        total_reward = 0
        while not done:
            # get the action
            env.render()
            action = agent([state,])[0].item()

            state, reward, done ,info = env.step(action)
            
            total_reward += reward
        rewards.append(total_reward)
        print('episode %d : reward: %d'%(i, total_reward))

    print('mean reward ', np.mean(rewards))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    #env = Monitor(env, './video', force=True)
    model = PGN(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(model, preprocessor= ptan.agent.float32_preprocessor, apply_softmax=True)

    #train(env, agent)
    #torch.save(model.state_dict(), 'model.pkl')
    model.load_state_dict(torch.load('model.pkl'))
    test(env,agent)

