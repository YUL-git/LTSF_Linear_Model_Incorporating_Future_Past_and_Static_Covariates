import os
import numpy as np
from tqdm.auto import tqdm
from .eval import validation
import torch
import torch.nn as nn

def train(model, optimizer, scheduler, train_loader, val_loader, device, config):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None

    for epoch in range(1, config['num_epochs']+1):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_loader)):
            X = tuple(map(lambda x: x.to(device), X))
            Y = Y.to(device)

            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, Y)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print('Model Saved')
            if epoch >= 10:
                model_save_path = config["model_save_path"]
                model_file_name = f"{epoch}_DLinear.pth"
                full_path = os.path.join(model_save_path, model_file_name)
                torch.save(best_model.state_dict(), full_path)
        scheduler.step()
        
    return best_model