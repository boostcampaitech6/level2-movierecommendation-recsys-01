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
import pytz as tz

import hydra
from omegaconf import DictConfig

from src.data.datasets import (
    DataPipeline, FMDataset,
    )
from src.trainer import Trainer
from src.utils import set_seed, create_data_path, save_submission

import torch
from torch.utils.data import DataLoader

import wandb

@hydra.main(config_path="./src/configs", config_name="train_config", version_base='1.3')
def main(args: DictConfig):
    # runname
    time_zone = tz.timezone('Asia/Seoul')
    now = dt.strftime(dt.now(time_zone), '%y%m%d-%H%M%S')
    runname = f"{args.model_name}_{now}"
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.submit_dir).mkdir(exist_ok=True, parents=True)

    # wandb init
    if args.wandb:
        print("wandb init...")
        wandb.init(project=args.project, config=dict(args), name=runname)

    # seed
    set_seed(args.seed)

    # create data_path
    data_path, train_path, valid_path = create_data_path(args)
    data_pipeline = DataPipeline(args)

    if (not os.path.exists(train_path)) or (not os.path.exists(valid_path)):
        Path(data_path).mkdir(exist_ok=True, parents=True)
        data = data_pipeline.preprocess_data()
        train_data, valid_data = data_pipeline.split_data(data)

        data_pipeline.save_data(train_data, train_path)
        data_pipeline.save_data(valid_data, valid_path)
    else:
        train_data = data_pipeline.load_data(train_path)
        valid_data = data_pipeline.load_data(valid_path)

    # ordinal encoding
    # cat_features = [name for name, options in args.feature_sets.items() if options == (1, 'C')]
    cat_features = ['user', 'item']
    train_data['X'] = data_pipeline.encode_categorical_features(train_data['X'], cat_features)
    valid_data['X'] = data_pipeline.encode_categorical_features(valid_data['X'], cat_features)
    
    # dataset
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Trainer 
    trainer = Trainer(args, data_pipeline.cat_features_size, runname)
    trainer.run(train_dataloader, valid_dataloader)

    # Load Best Model
    trainer.load_best_model()

    # Inference
    prediction = trainer.inference()
    prediction = data_pipeline.decode_categorical_features(prediction)
    save_submission(prediction, args, runname)
    
if __name__ == '__main__':
    main()
