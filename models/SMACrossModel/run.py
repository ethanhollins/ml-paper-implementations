import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm
from utils import load_price_data, calculate_distance, min_max_clipped, batch_df
from prepare import calculate_sma, get_labels
from networks import Agent


'''
Pre-processing
'''
SOURCE = "fxcm"
INSTRUMENT = "EUR_USD"
PERIOD = "M1"
start_date = datetime(2020,1,1)
end_date = datetime(2020,12,31)


class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # Return total size
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, index):
        # Return batch
        return self.data[:, index:index+1, :], self.labels[:, index:index+1, :]


def preprocess(device, batch_size=512):
    # Load Data
    data = load_price_data(SOURCE, INSTRUMENT, PERIOD, start_date, end_date)
    data = data[["midopen", "midhigh", "midlow", "midclose"]]

    # Calculate SMAs
    sma1 = calculate_sma(data, 20)
    sma2 = calculate_sma(data, 40)

    # Resize all to same size
    sma1 = sma1[sma1.shape[0]-sma2.shape[0]:]

    # Convert DataFrames from prices to distances
    sma1 = calculate_distance(sma1)
    sma2 = calculate_distance(sma2)

    # Get difference between SMAs
    X = sma1 - sma2

    # Batch data
    y = batch_df(get_labels(X), batch_size)
    X = batch_df(min_max_clipped(X), batch_size)

    # Split/shuffle batch data into training and testing sets
    test_split = int(X.shape[1] * 0.2)
    train_split = X.shape[1] - test_split

    train_indices = np.random.choice(np.arange(X.shape[1], dtype=int), train_split, replace=False)
    X_train = X[:, train_indices, :]
    y_train = y[:, train_indices, :]

    X_test = np.delete(X, train_indices, axis=1)
    y_test = np.delete(y, train_indices, axis=1)

    # Convert to PyTorch Tensors
    X_train = T.tensor(X_train, dtype=T.float).to(device)
    y_train = T.tensor(y_train, dtype=T.float).to(device)
    X_test = T.tensor(X_test, dtype=T.float).to(device)
    y_test = T.tensor(y_test, dtype=T.float).to(device)

    # Show data info
    print(f"Train: {X_train.shape}\nTest: {X_test.shape}")

    train_set = MyDataset(X_train, y_train)
    test_set = MyDataset(X_test, y_test)

    return train_set, test_set


def get_agent(device, state_space, n_actions):
    return Agent(device, state_space, n_actions)

    
def train(agent, train_set, ep_num):
    agent.net.train()
    for batch_idx, (data, labels) in enumerate(train_set):
        _, loss = agent.learn(data, labels)
        if batch_idx % 10 == 0:
            print("Train Ep {}: [{:06d}/{:06d} ({:.0f})%]\tLoss: {:.6f}".format(
                ep_num, batch_idx * len(data), len(train_set),
                100. * batch_idx / len(train_set.data[1]), loss.item()
            ), end='\r')

def test(agent, test_set):
    agent.net.eval()
    test_loss = 0
    correct = 0
    with T.no_grad():
        for data, labels in test_set:
            output, loss = agent.learn(data, labels)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_set)

    print("\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_set),
        100. * correct / len(test_set)
    ))


def run(train_set, test_set, agent, episodes=100, save=False):
    for ep_num in range(episodes):
        train(agent, train_set, ep_num)
        test(agent, test_set)

    if save:
        agent.save_models()

if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    train_set, test_set = preprocess(device, batch_size=1024)
    agent = get_agent(device, train_set.data.shape[2], train_set.labels.shape[2])
    run(train_set, test_set, agent, episodes=1)
