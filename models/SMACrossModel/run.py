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
start_date = datetime(2017,1,1)
end_date = datetime(2020,12,31)


def preprocess(batch_size=512):
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
    X_train = T.tensor(X_train, dtype=T.float)
    y_train = T.tensor(y_train, dtype=T.float)
    X_test = T.tensor(X_test, dtype=T.float)
    y_test = T.tensor(y_test, dtype=T.float)

    # Show data info
    print(f"Train: {X_train.shape}\nTest: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def get_agent(device, state_space, n_actions):
    return Agent(device, state_space, n_actions)


def main(episodes=100, save=False):
    input_space = 1
    action_space = 2

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = preprocess(batch_size=1024)
    agent = get_agent(device, input_space, action_space)

    batch_step = 500
    for ep_num in range(episodes):
        for batch_idx in range(0, X_train.shape[1], batch_step):
            _, train_loss = agent.train(
                X_train[:, batch_idx:batch_idx+batch_step], 
                y_train[:, batch_idx:batch_idx+batch_step]
            )
            if batch_idx % 10 == 0 or batch_idx + batch_step >= X_train.shape[1]:
                print("Train Batch {:03d}: [{:03d}/{:03d} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx, batch_idx, X_train.shape[1],
                    100. * batch_idx / X_train.shape[1], train_loss.item()
                ), end='\r')

        test_loss = 0
        correct = 0
        for batch_idx in range(0, X_test.shape[1], batch_step):
            batch_loss, batch_correct = agent.test(
                X_test[:, batch_idx:batch_idx+batch_step], 
                y_test[:, batch_idx:batch_idx+batch_step]
            )
            test_loss += batch_loss
            correct += batch_correct

        test_loss /= (X_test.shape[1] / batch_step)
        print("\n(Ep {}/{}) Test set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            ep_num, episodes, test_loss, correct, X_test.shape[0] * X_test.shape[1],
            100. * correct / (X_test.shape[0] * X_test.shape[1])
        ))

    if save:
        agent.save_models()

if __name__ == "__main__":
    main(episodes=200)
