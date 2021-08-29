import argparse
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Sequence(nn.Module):

    def __init__(self, device):
        super(Sequence, self).__init__()

        self.device = device

        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.fc1 = nn.Linear(51, 1)

    
    def forward(self, x, future=0):
        outputs = []
        h_t = T.zeros(x.size(0), 51, dtype=T.double).to(self.device)
        c_t = T.zeros(x.size(0), 51, dtype=T.double).to(self.device)
        h_t2 = T.zeros(x.size(0), 51, dtype=T.double).to(self.device)
        c_t2 = T.zeros(x.size(0), 51, dtype=T.double).to(self.device)

        for x_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(x_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.fc1(h_t2)
            outputs += [output]

        for _ in range(future): # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.fc1(h_t2)
            outputs += [output]

        outputs = T.cat(outputs, dim=1)
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # Set random seed to 0
    np.random.seed(0)
    T.manual_seed(0)
    # Load data and make training set
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    data = T.load("traindata.pt")
    input_ = T.from_numpy(data[3:, :-1]).to(device)
    target = T.from_numpy(data[3:, 1:]).to(device)
    test_input = T.from_numpy(data[:3, :-1]).to(device)
    test_target = T.from_numpy(data[:3, 1:]).to(device)
    # Build sequence
    seq = Sequence(device)
    seq.double()
    seq.to(device)
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # Begin training
    for i in range(opt.steps):
        print("STEP: ", i)
        def closure():
            optimizer.zero_grad()
            out = seq(input_)
            loss = criterion(out, target)
            print("Loss: ", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with T.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print("Test loss: ", loss.item())
            y = pred.detach().to("cpu").numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title("Predict future values for time sequences\n(Dashlines are predicted values)", fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input_.size(1)), yi[:input_.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input_.size(1), input_.size(1) + future), yi[input_.size(1):], color + ':', linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('./tmp/predict%d.pdf' % i)
        plt.close()


if __name__ == "__main__":
    main()
