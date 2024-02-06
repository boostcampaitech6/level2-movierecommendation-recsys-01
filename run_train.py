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
from datetime import datetime as dt
from src.data.datasets import (
    get_data, split_data, save_data, load_data, FMDataset,
    encode_data, decode_data, save_submission)
from src.trainer import Trainer
from src.utils import set_seed

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./src/configs", config_name="train_config", version_base='1.3')
def main(args: DictConfig):
    # runname
    now = dt.strftime(dt.now(), '%y%m%d-%H%M%S')
    runname = f"{args.model_name}_{now}"

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
