import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer():

    def __init__(self, device, buffer_size, input_dims, n_actions):
        self.buffer_size = buffer_size
        self.mem_cntr = 0

        self.state_mem = T.zeros((buffer_size, input_dims), dtype=T.float64).to(device)
        self.action_mem = T.zeros((buffer_size, n_actions), dtype=T.float64).to(device)
        self.reward_mem = T.zeros((buffer_size, 1), dtype=T.float64).to(device)
        self.next_state_mem = T.zeros((buffer_size, input_dims), dtype=T.float64).to(device)
        self.terminal_mem = T.zeros((buffer_size, 1), dtype=T.float64).to(device)

    
    def save_experience(self, state, action, reward, next_state, done):
        idx = self.mem_cntr % self.buffer_size
        
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.terminal_mem[idx] = done


    def sample_experience(self, batch_size):
        exp_count = min(self.mem_cntr, self.buffer_size)

        batch = np.random.choice(exp_count, batch_size)

        state = self.state_mem[batch]
        action = self.action_mem[batch]
        reward = self.reward_mem[batch]
        next_state = self.next_state_mem[batch]
        done = self.terminal_mem[batch]

        return state, action, reward, next_state, done


    def is_sample_ready(self, batch_size):
        return self.mem_cntr > batch_size


class DDQNNetwork(nn.Module):

    def __init__(self, input_dims, n_actions):
        super(DDQNNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions

        self.local_fc1 = nn.Linear(input_dims, 512)
        self.local_fc2 = nn.Linear(512, 512)
        self.local_q = nn.Linear(512, n_actions)

        self.target_fc1 = nn.Linear(input_dims, 512)
        self.target_fc2 = nn.Linear(512, 512)
        self.target_q = nn.Linear(512, n_actions)

    
    def forward(self, x):
        local_x = F.relu(self.local_fc1(x))
        local_x = F.relu(self.local_fc2(local_x))
        local_q = self.local_q(local_x)
        max_action_indicies = local_q.argmax(1)
        
        target_x = F.relu(self.target_fc1(x))
        target_x = F.relu(self.target_fc2(target_x))
        target_q = self.target_q(target_x)

        return target_q.gather(1, max_action_indicies)


    def save_checkpoint(self):
        T.save(self.state_dict(), "tmp/ddqn.pt")

    
    def load_checkpoint(self):
        self.load_state_dict(T.load("tmp/ddqn.pt"))


class DDQNAgent():

    def __init__(self, device, state_space, action_space, buffer_size):
        
        self.net = DDQNNetwork(state_space, action_space)
        self.memory = ReplayBuffer(device, buffer_size, state_space, action_space)

    
    def select_action(self, state):
        return

    
    def train(self, optimizer, state):
        self.net.train()

        if not self.memory.is_sample_ready():
            return

        state, action, reward, next_state, done = self.memory.sample_experience(self.batch_size)

        optimizer.zero_grad()

        target_q = reward + T.mul(self.gamma * self.net(next_state), 1 - done)
        old_q = self.net(state).gather(1, action.long())

        loss = F.smooth_l1_loss(old_q, target_q)
        loss.backward()
        optimizer.step()

        # epsilon decay
        

    
    def eval(self):
        self.net.eval()