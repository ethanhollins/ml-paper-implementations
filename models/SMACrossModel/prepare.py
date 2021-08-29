import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from utils import calculate_distance, min_max


'''
Pre-processing
'''

# Calculate labels

def get_labels(data):
    '''
    Args:
        - data (ndarray): Array of distance between price values

    Create label array where the target action is equal to 1 if
    SMA1 has a distance greater than 0 to SMA2 and 0 for the
    contrary.
    '''

    return (data >= 0).astype(np.int32)


# Pre-process training data

def calculate_sma(df, length, type_="close"):
    '''
    Args
        - data (ndarray): Array of price data
        - length (int): Length of SMA
        - type_ (str): Prices to use for SMA
    
    Calculates Simple Moving Average (SMA)
    '''
    
    if type_ == "open":
        data = df.values[:, 0]
    elif type_ == "high":
        data = df.values[:, 1]
    elif type_ == "low":
        data = df.values[:, 2]
    elif type_ == "close":
        data = df.values[:, 3]

    result = np.zeros((data.shape[0]-length))
    for i in range(length, data.shape[0]):
        result[i-length] = data[i-length:i].mean()
        if result[i-length] == np.nan:
            print(data[i-length:i])

    return pd.DataFrame(data=result, index=df.index[length:], columns=["items"])


def preprocess_data(data):
    ''' 
    Args
        - data (ndarray): Array of price data

    Pre-process data for training 
    '''
    
    data = calculate_distance(data)
    data = min_max(data)
    return data

