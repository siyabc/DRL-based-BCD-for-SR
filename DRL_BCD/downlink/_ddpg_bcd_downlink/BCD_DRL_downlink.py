#ddpg 来实现论文

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from data_gen.BCD_WSRM import iteration_for_subproblem

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# class ValueNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_size ,init_w = 3e-3):
#         super(ValueNetwork, self).__init__()
#
#         self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.linear3 = nn.Linear(hidden_size, 1)
#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_layers=3, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        layers = []
        input_size = num_inputs + num_actions

        # 添加第一层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # 添加中间层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # 添加最后一层
        layers.append(nn.Linear(hidden_size, 1))

        # 将所有层组合成一个 Sequential 模块
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.model(x)  # 使用 nn.Sequential 定义的模型
        return x

# class PolicyNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_size, num_layers = 3,
#                  init_w = 3e-3):
#         super(PolicyNetwork, self).__init__()
#         self.num_actions = num_actions
#         self.linear1 = nn.Linear(num_inputs, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.linear3 = nn.Linear(hidden_size, num_actions)
#
#     def forward(self, x):
#             x = F.relu(self.linear1(x))
#             x = F.relu(self.linear2(x))
#             x = F.relu(self.linear3(x))
#             return x

    #======
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_layers=3,
                 init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        layers = []
        layers.append(nn.Linear(num_inputs, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):  # 创建 num_layers - 1 个隐藏层
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))  # 输出层
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
            x = self.model(x)
            return x
    def get_action(self, state):
        noise_scale = 0.001
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        action += noise_scale * np.random.rand(self.num_actions)
        return action


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def plot(frame_idx, rewards):
    plt.figure(figsize=(5,5))
    plt.subplot(111)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 20
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 5000 # 1500,3000,5000，8000
        self.value_lr = 1*1e-6 #1e-4, 1e-5, 1e-6,5e-7
        self.policy_lr = 1*1e-6

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

def sr_step(o, a, label):
    G = o[0:9].reshape(3, 3)
    w = o[9:12].reshape(3, 1)
    sigma = o[12:15].reshape(3, 1)
    p_bar = o[15]
    gamma_star = label
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    # print(np.dot(v,np.ones((1, 3))))
    B = F+1/p_bar*np.dot(v,np.ones((1, 3)))
    # y = 1/(1/np.exp(a)+1)
    # b = w[:,0]*y
    b = w[:, 0] * a
    til_gamma = iteration_for_subproblem(B, b)
    gamma = np.exp(til_gamma)

    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    reward = - (np.abs(obj_updated-obj_star))**2
    # reward = - np.linalg.norm(gamma - gamma_star, 1)
    # print("a:", a)
    # print("gamma:", gamma)
    # print("gamma_star:", gamma_star)
    # print("reward:", reward)
    o2 = np.append(o[:-3], gamma)
    d = False
    if reward >= -1e-3:
        d = True
    return o2, reward, d


def traning():
    sr_data = list()
    sr_target = list()
    with open("data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sr_data.append(list(map(float, line[:-3])))
            sr_target.append(list(map(float, line[-3:])))

    action_space = 2
    # ou_noise = OUNoise(action_space)

    state_dim = 19
    action_dim = 3
    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)

    max_frames = 150
    #2500,3500,4500,5500
    max_steps = 4500
    frame_idx = 0
    # rewards = []
    replay_size = 0
    start_update = 20

    rewardList = []
    err_list = []
    stop_step_list = []
    outlier_list = []

    while frame_idx < max_frames:
        print("frame_idx:", frame_idx)
        # state = env.reset()
        o_init = np.append(np.array(sr_data[frame_idx]), np.random.rand(3) * 0.1)
        label = np.array(sr_target[frame_idx])

        state = o_init
        # ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            # env.render()
            action = ddpg.policy_net.get_action(state)
            # action = ou_noise.get_action(action, step)
            # next_state, reward, done, _ = env.step(action)
            next_state, reward, done = sr_step(state, action, label)
            # print("action:", len(action))
            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            # if frame_idx >= start_update and len(ddpg.replay_buffer) > replay_size:
            if frame_idx >= start_update:
                ddpg.ddpg_update()


            state = next_state
            episode_reward += reward
            if done:
                break

        gamma = state[-3:]
        err = np.linalg.norm(gamma - label)
        print('Episode:', frame_idx, 'gamma:', gamma, 'label:', label, '==========', 'Reward:', episode_reward, 'err:', err,
              'j:', step)


        frame_idx += 1
        print()

        rewardList.append(episode_reward)
        err_list.append(err)
        stop_step_list.append(step)
        if frame_idx > 50 and step >= 1000:
            outlier_list.append(frame_idx)

    print("rewardList:", rewardList)
    print("err_list:", err_list)
    print("stop_step_list:", stop_step_list)

    # print("outlier_list:", outlier_list)

    print("outlier_list:", outlier_list)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("ddpg_Reward", fontsize=10)

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(len(err_list)), err_list)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("SINR error", fontsize=10)

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(stop_step_list)), stop_step_list)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Iteration steps", fontsize=10)

    plt.savefig("res_v1.pdf")
    plt.show()
    # plot(frame_idx, rewards)

    with open('res_reward_ddpg.csv', 'w', newline='') as file:
        for i in range(max_frames):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array([rewardList[i]])
            mywriter.writerow(a)
    with open('res_err_ddpg.csv', 'w', newline='') as file:
        for i in range(max_frames):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array([err_list[i]])
            mywriter.writerow(a)
    with open('res_step_ddpg.csv', 'w', newline='') as file:
        for i in range(max_frames):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array([stop_step_list[i]])
            mywriter.writerow(a)



if __name__ == '__main__':
    traning()
