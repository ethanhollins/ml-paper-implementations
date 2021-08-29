import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class Network():

    def __init__(
        self, input_size, n_actions, lr=1e-2,
        conv1_size=32, conv1_kernal_size=8, conv1_stride=2, 
        conv2_size=64, conv2_kernal_size=4, conv2_stride=2,
        conv3_size=64, conv3_kernal_size=3, conv3_stride=1,
        lstm1_size=512
    ):

        self.conv1 = nn.Conv2d(input_size, conv1_size, conv1_kernal_size, conv1_stride)
        self.conv2 = nn.Conv2d(conv1_size, conv2_size, conv2_kernal_size, conv2_stride)
        self.conv3 = nn.Conv2d(conv2_size, conv3_size, conv3_kernal_size, conv3_stride)

        self.lstm1 = nn.LSTMCell(conv3_size, lstm1_size)
        self.q = nn.Linear(lstm1_size, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        
    def forward(self, state):
        ''' Forward step network '''
        
        # Conv Layer 1
        x1 = self.conv1(state)
        x1 = F.relu(x1)
        # Conv Layer 2
        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        # Conv Layer 3
        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        # LSTM Layer 1
        x4 = self.lstm1(x3)
        x4 = F.relu(x4)
        # Fully Connected Layer
        q = self.q(x4)
        return q


    def backward(self, prediction, target):
        ''' Backpropagate network '''
        self.optimizer.zero_grad()

        loss = F.smooth_l1_loss(prediction, target)
        loss.backward()

        self.optimizer.step()

    
    def save_checkpoint(self):
        return

    
    def load_checkpoint(self):
        return
