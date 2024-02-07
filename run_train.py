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
from pathlib import Path
from datetime import datetime as dt

import hydra
from omegaconf import DictConfig

from src.data.datasets import (
    get_data, split_data, save_data, load_data, FMDataset,
    encode_data, decode_data, save_submission)
from src.trainer import Trainer
from src.utils import set_seed, create_data_path

import torch
from torch.utils.data import DataLoader

@hydra.main(config_path="./src/configs", config_name="train_config", version_base='1.3')
def main(args: DictConfig):
    # runname
    now = dt.strftime(dt.now(), '%y%m%d-%H%M%S')
    runname = f"{args.model_name}_{now}"
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.submit_dir).mkdir(exist_ok=True, parents=True)

    # seed
    set_seed(args.seed)

    # create data_path
    data_path, train_path, valid_path = create_data_path(args)

    if not os.path.exists(data_path):
        Path(data_path).mkdir(exist_ok=True, parents=True)
        data = get_data()
        train_data, valid_data = split_data(data)

        save_data(train_data, train_path)
        save_data(valid_data, valid_path)
    else:
        train_data = load_data(train_path)
        valid_data = load_data(valid_path)

    # ordinal encoding
    cat_features = ['user', 'item']
    train_data['X'][cat_features], oe, cat_features_size = encode_data(train_data['X'][cat_features], train=True)
    valid_data['X'][cat_features], _, _ = encode_data(valid_data['X'][cat_features], train=False, ordinal_encoder=oe)
    
    # dataset
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Trainer 
    trainer = Trainer(args, cat_features_size, runname)
    trainer.run(train_dataloader, valid_dataloader)

    # Load Best Model
    trainer.load_best_model()

    # Inference
    prediction = trainer.inference()
    prediction = decode_data(oe, prediction)
    save_submission(prediction, args, runname)
    
if __name__ == '__main__':
    main()
