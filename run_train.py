'''
run_train.py

- args
- set seed
- wandb setting
- load data
- dataloader
- train
- evaluation
'''
import os
from easydict import EasyDict
from src.data.datasets import (
    get_data, split_data, save_data, load_data, FMDataset)
from src.trainer import Trainer
from src.utils import set_seed

import torch
from torch.utils.data import DataLoader


def main():
    # args 
    args = EasyDict({
        "train_name": 'train_data.pickle',
        "valid_name": 'valid_data.pickle',
        "batch_size": 1024,
        "model_name": "DeepFM",
        "epochs": 10,
        "emb_dim": 64,
        "lr": 0.001,
        "optimizer": "adamw",
        "seed": 42,
    })
    
    # seed
    set_seed(args.seed)

    if not os.path.exists(args.train_name):
        data = get_data()
        train_data, valid_data = split_data(data)

        save_data(train_data, args.train_name)
        save_data(valid_data, args.valid_name)
    else:
        train_data = load_data(args.train_name)
        valid_data = load_data(args.valid_name)
    
    # dataset
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Trainer 
    trainer = Trainer(args, train_data["cat_features_size"])
    trainer.run(train_dataloader, valid_dataloader)
    
    
if __name__ == '__main__':
    main()
