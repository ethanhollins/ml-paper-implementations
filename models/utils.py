import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
PRICES_PATH = os.path.join(DATA_PATH, "prices")

def load_price_data(source, instrument, period, start, end):
    '''
    Args
        - source (str): Source provider of price data (i.e. broker)
        - instrument (str): Instrument of price data
        - period (str): Time period of price data
        - start (datetime): Start date of data collection
        - end (datetime): End date of data collection

    Collects local price data and concatenates into DataFrame
    '''
    print(f"Collecting Data {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")

    frags = []
    file_schema, file_ext = '%Y%m%d', '.csv.gz'
    names = [
        "timestamp",
        "askopen", "askhigh", "asklow", "askclose",
        "midopen", "midhigh", "midlow", "midclose",
        "bidopen", "bidhigh", "bidlow", "bidclose"
    ]
    while start < end:
        file_path = os.path.join(PRICES_PATH, source, instrument, period, start.strftime(file_schema) + file_ext)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, names=names, compression="gzip")
            df.index = df.index.astype(np.int64)
            frags.append(df)
        start += timedelta(days=1)
            
    if len(frags) > 0:
        return pd.concat(frags)
    else:
        return None

    
def calculate_distance(data):
    ''' 
    Args:
        - data (ndarray): Array of price data

    Calculate the distance between every element and its 
    previous element
    '''
    # shifted = np.concatenate(( np.zeros([1] + list(data.shape[1:])), data[:-1] ))
    shifted = data[:-1]
    shifted.index = data[1:].index
    return (data[1:] - shifted) * 10e4


def min_max(data):
    '''
    Args:
        - data (ndarray): Array of distance between price values
    
    Min max scale data
    '''

    scaler = MinMaxScaler((-1,1))
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(data=scaled_data, index=data.index, columns=data.columns)

    return data


def min_max_clipped(data, num_stds=3):
    '''
    Args:
        - data (ndarray): Array of distance between price values
    
    Clipped by a number of standard deviations from median and min max scaled
    '''

    median = data.median().mean()
    std = data.std().mean()
    data.clip(lower=median-std*num_stds, upper=median+std*num_stds, inplace=True)
    scaler = MinMaxScaler((-1,1))
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(data=scaled_data, index=data.index, columns=data.columns)

    return data


def batch_df(df, batch_size, padding=0):

    result = np.ones((batch_size, math.ceil(df.shape[0] / batch_size), df.shape[1])) * padding
    for i in range(math.ceil(df.shape[0] / batch_size)):
        data = df.values[i*batch_size:(i+1)*batch_size]
        result[:data.shape[0], i] = data

    return result

# TESTING 
# df = load_price_data("fxcm", "EUR_USD", "M1", datetime(2020,1,1), datetime(2020,3,1))
# print(f"DF               :\n{df.head(5)}")
# df_dist = calculate_distance(df)
# print(f"DF Dist          :\n{df_dist.head(5)}")
# df_min_max = min_max(df_dist)
# print(f"DF MinMax        :\n{df_min_max.head(5)}")
# df_min_max_clipped = min_max_clipped(df_dist)
# print(f"DF MinMax Clipped:\n{df_min_max_clipped.head(5)}")
