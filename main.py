import DLinear
from DLinear.utils import seed_everything
from DLinear.train import train
from DLinear.predict import inference
from DLinear.model import DLinearModel
from DLinear.visualizer import plot_sales_and_predictions

import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def main(config):
    with open('./data/train_scale_min_dict.pickle', 'rb') as f:
        scale_min_dict = pickle.load(f)

    with open('./data/train_scale_max_dict.pickle', 'rb') as f:
        scale_max_dict = pickle.load(f)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    seed_everything(config)
    train_data, sales_data, submit_data = DLinear.read_data(config)

    train_x_past, train_x_future, train_x_static, train_x_target = DLinear.make_train_data(train_data, sales_data, config)
    test_x_past, test_x_future, test_x_static = DLinear.make_test_data(train_data, sales_data, submit_data, config)

    train_x_past, val_x_past = DLinear.split_data(train_x_past)
    train_x_future, val_x_future = DLinear.split_data(train_x_future)
    train_x_static, val_x_static = DLinear.split_data(train_x_static)
    train_x_target, val_x_target = DLinear.split_data(train_x_target)

    train_dataset = DLinear.CustomDataset(train_x_past, train_x_future, train_x_static, train_x_target)
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)

    val_dataset = DLinear.CustomDataset(val_x_past, val_x_future, val_x_static, val_x_target)
    val_loader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle=False, num_workers=0)

    test_dataset = DLinear.CustomDataset(test_x_past, test_x_future, test_x_static, None)
    test_loader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=False, num_workers=0)

    model = DLinearModel(config, shared_weights=False, const_init=True)
    lambda1 = lambda epoch: 0.95 ** epoch
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    infer_model = train(model, optimizer, scheduler, train_loader, val_loader, device, config)

    pred = inference(infer_model, test_loader, device)
    for idx in range(len(pred)):
        pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    pred = np.round(pred, 0).astype(int)
    submit_data.iloc[:, 1:] = pred
    submit_data.to_csv(config["submision_path"], index=False)

    ## visualize
    past_sales = pd.read_csv("./data/train.csv").drop(columns=['ID','대분류','중분류','소분류','브랜드','제품'])
    past_sales_data = past_sales.loc[:, "2023-01-01":"2023-04-04"]

    for i in range(0,10):
        plot_sales_and_predictions(past_sales_data[i], pred[i], start_date="2023-01-01", end_date="2023-04-26")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    train_data, sales_data 모두 이미 scale이 진행된 데이터
    """
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--input_chunk_length", type=int, default=90)
    parser.add_argument("--output_chunk_length", type=int, default=21)
    parser.add_argument("--input_dim", type=int, default=5)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--future_cov_dim", type=int, default=3)
    parser.add_argument("--static_cov_dim", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", default="./best_model/")
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--train_file", default="train_data.csv")
    parser.add_argument("--sales_file", default="sales_data.csv")
    parser.add_argument("--submit_file", default="sample_submission.csv")
    parser.add_argument("--submission_path", default="./submission/DLinear.csv")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "input_chunk_length": args.input_chunk_length,
        "output_chunk_length": args.output_chunk_length,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "future_cov_dim": args.future_cov_dim,
        "static_cov_dim": args.static_cov_dim,
        "kernel_size": args.kernel_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "model_save_path": args.model_save_path,
        "data_path": args.data_path,
        "train_file": args.train_file,
        "sales_file": args.sales_file,
        "submit_file": args.submit_file,
        "submission_path": args.submission_path
    }
    print(config)
    main(config)