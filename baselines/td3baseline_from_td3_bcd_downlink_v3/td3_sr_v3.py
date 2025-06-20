from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import core_sr_v3 as core_sr


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core_sr.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core_sr.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core_sr.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class TD3:
    def __init__(self, obs_dim, act_dim,nn_dim, actor_critic=core_sr.MLPActorCritic,
                 replay_size=int(1e6), gamma=0.9, polyak=0.995, pi_lr=1e-5, q_lr=1e-5,
                 act_noise=0.05, target_noise=0.05, noise_clip=0.5, policy_delay=20,hidden_sizes=(64,64)):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,size=replay_size)

        self.ac = actor_critic(nn_dim, act_dim, hidden_sizes=hidden_sizes)
        self.ac_targ = deepcopy(self.ac)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        # # Experience buffer
        # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = o[:,16:]
        o2 = o2[:,16:]
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -1, 1)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # print("--d:",d)
            # print("==q1:",q1_pi_targ)
            # print("==q2:",q2_pi_targ)

            # print("==q:", q_pi_targ)

            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        o = o[:,16:]    # o_for_act
        a = self.ac.pi(o)
        q1_pi = self.ac.q1(o, self.ac.pi(o))
        # print("==a_pi:", a.reshape(30))
        # print("q1_pi:", q1_pi)



        return -q1_pi.mean()

    def update(self, batch_size, repeat_times):
        for i in range(int(repeat_times)):
            data = self.replay_buffer.sample_batch(batch_size)
            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer.zero_grad()
            loss_q = self.compute_loss_q(data)
            loss_q.backward()
            self.q_optimizer.step()

            # Possibly update pi and target networks
            if i % self.policy_delay == 0:

                # Freeze Q-networks so you don't waste computational effort
                # computing gradients for them during the policy learning step.
                for p in self.q_params:
                    p.requires_grad = False

                # Next run one gradient descent step for pi.
                self.pi_optimizer.zero_grad()
                loss_pi = self.compute_loss_pi(data)
                # print("loss_pi:", loss_pi)

                loss_pi.backward()
                self.pi_optimizer.step()

                for name, param in self.ac.pi.named_parameters():
                    if name == "pi.2.weight":  # 检查参数名称是否是 'pi.2.weight'
                        print(f"Parameter Name: {name}")
                        print(param)

                # for name, param in self.ac.pi.named_parameters():
                #     if name == "pi.2.weight":  # 检查参数名称是否是 'pi.2.weight'
                #         print(f"Parameter Name: {name}")
                #         if param.grad is not None:
                #             print("Gradient:")
                #             print(param.grad)  # 打印该参数的梯度
                #         else:
                #             print("No gradient for this parameter")

                        # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for p in self.q_params:
                    p.requires_grad = True

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o_for_act, noise_scale):
        # for name, param in self.ac.pi.named_parameters():
        #     if name == "pi.2.weight":  # 检查参数名称是否是 'pi.2.weight'
        #         print(f"Parameter Name: {name}")
        #         print(param)
        a = self.ac.act(torch.as_tensor(o_for_act, dtype=torch.float32))
        # print("a1:", a)
        a += noise_scale * np.random.rand(self.act_dim)
        # print("a2:", a)

        # return np.clip(a, 0, 1)
        return a
