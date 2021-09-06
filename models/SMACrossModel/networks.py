import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, device, input_size, n_actions, lr, lstm1_size):

        super(Network, self).__init__()

        self.device = device

        self.input_size = input_size
        self.n_actions = n_actions

        self.lstm1_size = lstm1_size
        
        self.lstm1 = nn.LSTMCell(self.input_size, self.lstm1_size)
        self.fc1 = nn.Linear(self.lstm1_size, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.to(self.device)

        
    def forward(self, x):
        ''' Forward step network '''
        output = T.zeros((x.shape[0], x.shape[1], self.n_actions)).to(self.device)
        # LSTM Layer 1
        h_t = Variable(T.zeros(x.size(1), self.lstm1_size)).to(self.device) # hidden state
        c_t = Variable(T.zeros(x.size(1), self.lstm1_size)).to(self.device) # internal state
        for t in range(x.shape[0]):
            h_t, c_t = self.lstm1(x[t], (h_t, c_t))
            result = F.log_softmax(self.fc1(h_t), dim=1)
            output[t] = result

        # Fully Connected Layer
        return output

    
    def save_checkpoint(self):
        ''' Save Checkpoint File '''
        T.save(self.state_dict(), "./tmp/net.pt")

    
    def load_checkpoint(self):
        ''' Load Checkpoint File '''
        return


class Agent():

    def __init__(self, device, state_space, n_actions, lr=1e-2, lstm1_size=32):

        self.state_space = state_space
        self.n_actions = n_actions

        self.net = Network(device, state_space, n_actions, lr, lstm1_size)


    def train(self, state, labels):
        state, labels = state.to(self.net.device), labels.to(self.net.device)

        self.net.optimizer.zero_grad()
        output = self.net(state)
        loss = F.smooth_l1_loss(output, labels)
        loss.backward()
        self.net.optimizer.step()

        return output, loss


    def test(self, state, labels):
        state, labels = state.to(self.net.device), labels.to(self.net.device)
        
        self.net.eval()

        with T.no_grad():
            output = self.net(state)
            loss = F.smooth_l1_loss(output, labels)

            test_loss = loss.item()
            pred = output.argmax(dim=2, keepdim=True)
            correct = pred.eq(labels.argmax(dim=2, keepdim=True)).sum().item()

            return test_loss, correct


    def save_models(self):
        ''' Save Network Models '''
        self.net.save_checkpoint()

    
    def load_models(self):
        ''' Load Network Models '''
        self.net.load_checkpoint()
