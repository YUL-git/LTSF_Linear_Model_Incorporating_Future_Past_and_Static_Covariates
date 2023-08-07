import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


def read_data(config):
    path = config["data_path"]
    train_file_path = config["train_file"]
    test_file_path = config["sales_file"]
    submit_file_path = config["submit_file"]

    train_path = os.path.join(path, train_file_path)
    test_path = os.path.join(path, test_file_path)
    submit_path = os.path.join(path, submit_file_path)

    train_data = pd.read_csv(train_path)
    sales_data = pd.read_csv(test_path)
    submit = pd.read_csv(submit_path)

    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류', '소분류', '브랜드']

    for col in categorical_columns:
        label_encoder.fit(train_data[col])
        train_data[col] = label_encoder.transform(train_data[col])
    
    return train_data, sales_data, submit

def make_train_data(
        train_data, sales_data, config
):
    """
    x_past = (샘플 개수, input_chunk_length, target(판매량)+past_covariates(일매출)+historical_future_covariates(년+월+일)) => (샘플 개수, 90,5)
    x_future = (샘플 개수, input_chunk_length, future_covariates(년+월+일)) => (샘플 개수, 90,3)
    x_static = (샘플 개수, static_covariates(대분류+중분류+소분류+브랜드)) => (샘플 개수, 4)
    y_target = (샘플 개수, output_chunk_length, target(판매량)) => (샘플 개수, 21, 1)
    """
    input_chunk_length = config["input_chunk_length"]
    output_chunk_length=21
    num_rows = len(train_data)
    window_size = input_chunk_length + output_chunk_length
    time_lenght = 459

    x_past = np.empty((num_rows * (time_lenght - window_size + 1), input_chunk_length, 5))
    x_future = np.empty((num_rows * (time_lenght - window_size + 1), output_chunk_length, 3))
    x_static = np.empty((num_rows * (time_lenght - window_size + 1), 4))
    y_target = np.empty((num_rows * (time_lenght - window_size + 1), output_chunk_length))

    date_index = pd.DatetimeIndex(sales_data.transpose().index)

    for i in tqdm(range(num_rows)):
        #
        sales_volume =  np.array(train_data.iloc[i, 4:])
        daily_sales = np.array(sales_data.iloc[i, :])
        static_cov = np.array(train_data.iloc[i, :4])

        for j in range(time_lenght - window_size + 1):
            target_window = sales_volume[j : j + window_size]
            past_cov_window = daily_sales[j : j + window_size]
            date_window = date_index[j : j + window_size]

            past_data = np.column_stack((
                target_window[:input_chunk_length],
                past_cov_window[:input_chunk_length],
                date_window.year[:input_chunk_length],
                date_window.month[:input_chunk_length],
                date_window.day[:input_chunk_length]
            ))

            future_data = np.column_stack((
                date_window.year[input_chunk_length:],
                date_window.month[input_chunk_length:],
                date_window.day[input_chunk_length:]
            ))

            x_past[i * (time_lenght - window_size + 1) + j] = past_data
            x_future[i * (time_lenght - window_size + 1) + j] = future_data
            x_static[i * (time_lenght - window_size + 1) + j] = static_cov
            y_target[i * (time_lenght - window_size + 1) + j] = target_window[input_chunk_length:]
    
    return x_past, x_future, x_static, y_target

def make_test_data(
        train_data, sales_data, submit_data, config):
    input_chunk_length = config["input_chunk_length"]
    output_chunk_length = config["output_chunk_length"]
    
    num_rows = len(train_data)
    past_date_index =  pd.DatetimeIndex(sales_data.transpose().index)
    future_date_index = pd.DatetimeIndex(submit_data.iloc[:,1:].transpose().index)

    x_past = np.empty((num_rows, input_chunk_length, 5))
    x_future = np.empty((num_rows, output_chunk_length, 3))
    x_static = np.empty((num_rows, 4))

    for i in tqdm(range(num_rows)):
        sales_volume =  np.array(train_data.iloc[i, -input_chunk_length:])
        daily_sales = np.array(sales_data.iloc[i, -input_chunk_length:])
        past_year = past_date_index.year[-input_chunk_length:]
        past_month = past_date_index.month[-input_chunk_length:]
        past_day = past_date_index.day[-input_chunk_length:]
        #
        future_year = future_date_index.year
        future_month = future_date_index.month 
        future_day = future_date_index.day
        #
        static_cov = np.array(train_data.iloc[:, :4])
        
        past_data = np.column_stack((
            sales_volume,
            daily_sales,
            past_year,
            past_month,
            past_day
        ))


        future_data = np.column_stack((
            future_year,
            future_month,
            future_day
        ))

        x_past[i] = past_data
        x_future[i] = future_data
        x_static[i] = static_cov

    return x_past, x_future, x_static


################################################################################################
################################################################################################

def split_data(data, test_size=0.2):
    data_len = len(data)
    train_data = data[:-int(data_len*test_size)]
    val_data = data[-int(data_len*test_size):]
    return train_data, val_data

class CustomDataset(Dataset):
    def __init__(self, x_past, x_future, x_static, Y):
        self.x_past = x_past
        self.x_future = x_future
        self.x_static = x_static
        self.Y = Y

    def __len__(self):
        return len(self.x_past)
    
    def __getitem__(self, idx):
        if self.Y is not None:
            return torch.Tensor(self.x_past[idx]), torch.Tensor(self.x_future[idx]), torch.Tensor(self.x_static[idx]), torch.Tensor(self.Y[idx])
        return torch.Tensor(self.x_past[idx]), torch.Tensor(self.x_future[idx]), torch.Tensor(self.x_static[idx])

