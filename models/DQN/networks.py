import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer():

    '''
    Replay memory buffer for sampling previous events
    '''

    def __init__(self, state_space, max_memory_size):
        self.state_space = state_space
        self.mem_size = max_memory_size
        self.mem_cntr = 0

        self.state_mem = T.zeros(self.mem_size, *self.state_space, dtype=T.float)
        self.next_state_mem = T.zeros(self.mem_size, *self.state_space, dtype=T.float)
        self.action_mem = T.zeros(self.mem_size, 1, dtype=T.float)
        self.reward_mem = T.zeros(self.mem_size, 1, dtype=T.float)
        self.terminal_mem = T.zeros(self.mem_size, 1, dtype=T.int)

    
    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_cntr  % self.mem_size

        self.state_mem[idx] = state
        self.next_state_mem[idx] = next_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = done

    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        next_states = self.next_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]

        return states, actions, rewards, next_states, dones


class DQNLayer(nn.Module):

    '''
    Neural Net with 2 fully connected layers
    '''

    def __init__(self, 
        input_dims, n_actions, fc1_dims=256, fc2_dims=256, 
        lr=1e-2, name="dqn", chkpt_dir="tmp"
    ):
        super(DQNLayer, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".chkpt")

        self.fc1 = nn.Linear(self.input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    
    def forward(self, state):
        action_value = self.fc1(state)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class DQNAgent():

    '''
    Agent for performing evaluation, learning steps
    '''

    def __init__(self,
        state_space, action_space, max_memory_size, batch_size, 
        gamma=0.9, lr=1e-2, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99
    ):
        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # DQN
        self.dqn = DQNLayer(state_space, action_space, lr=lr)
        self.dqn.to(self.device)

        # Create memory
        self.memory = ReplayBuffer(state_space, max_memory_size)
        self.batch_size = batch_size

        # Learning Parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    
    def select_action(self, state):
        ''' Select Epsilon-greedy action '''

        state = state.to(self.device)
        if np.random.random() < self.epsilon:
            return T.tensor([[np.random.choice(self.action_space)]])
        else:
            return T.argmax(self.dqn(state)).unsqueeze(0).unsqueeze(0).cpu()

    
    def learn(self):
        # Wait for batch_size number of memories saved
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample from Replay Buffer
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Backpropergation
        self.dqn.optimizer.zero_grad()
        # Q-Learning is Q*(S, A) <- r + γ max_a Q(S', a)
        q_hat = reward + T.mul(self.gamma * self.dqn(next_state).max(1).values.unsqueeze(1), 1 - done)
        q_old_policy = self.dqn(state).gather(1, action.long())
        loss = F.mse_loss(q_old_policy, q_hat)

        loss.backward()
        self.dqn.optimizer.step()

        # Degrade epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    
    def save_models(self):
        print(".... Saving model ....")
        self.dqn.save_checkpoint()

    
    def load_models(self):
        print(".... Loading model ....")
        self.dqn.load_checkpoint()
