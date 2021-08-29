import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, device, input_size, n_actions, lr, lstm1_size):

        super(Network, self).__init__()

        self.input_size = input_size
        self.n_actions = n_actions

        self.lstm1_size = lstm1_size
        self.lstm1 = nn.LSTM(input_size, self.lstm1_size)
        self.fc1 = nn.Linear(lstm1_size, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.to(device)

        
    def forward(self, x):
        ''' Forward step network '''
        
        # LSTM Layer 1
        h_0 = Variable(T.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(T.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        _, (hn, cn) = self.lstm1(x, (h_0, c_0))
        hn = hn.view(-1, self.lstm1_size)
        x = F.relu(hn)
        x = self.fc1(x)
        # Fully Connected Layer
        output = F.log_softmax(x, dim=1)
        return output

    
    def save_checkpoint(self):
        ''' Save Checkpoint File '''
        T.save(self.state_dict(), "./tmp/net.pt")

    
    def load_checkpoint(self):
        ''' Load Checkpoint File '''
        return


class Agent():

    def __init__(self, device, state_space, n_actions, lr=1e-2, lstm1_size=256):

        self.state_space = state_space
        self.n_actions = n_actions

        self.net = Network(device, state_space, n_actions, lr, lstm1_size)


    def learn(self, state, labels):

        self.net.optimizer.zero_grad()
        output = self.net(state)
        if self.net.training:
            loss = F.smooth_l1_loss(output, labels)
        else:
            loss = F.smooth_l1_loss(output, labels, reduction="sum")
        loss.backward()
        self.net.optimizer.step()

        return output, loss


    def save_models(self):
        ''' Save Network Models '''
        self.net.save_checkpoint()

    
    def load_models(self):
        ''' Load Network Models '''
        self.net.load_checkpoint()
