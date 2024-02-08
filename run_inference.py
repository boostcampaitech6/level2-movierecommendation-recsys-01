'''
run_inference.py

- args
- set seed
- load data
- dataloader
- inference 
'''
import hydra
from omegaconf import DictConfig

from src.data.datasets import (
    DataPipeline, FMDataset,
    )
from src.inference import Inferencer
from src.utils import set_seed, create_data_path, save_submission

import torch
from torch.utils.data import DataLoader

@hydra.main(config_path="./src/configs", config_name="inference_config", version_base='1.3')
def main(args: DictConfig):
    # create data_path
    data_path, train_path, valid_path = create_data_path(args)
    data_pipeline = DataPipeline(args)

    # 데이터 이미 있다고 가정
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
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2048, shuffle=True)

    # Load Best Model
    inferencer = Inferencer(args, data_pipeline.cat_features_size, train_dataloader, valid_dataloader)

    # Inference
    prediction = inferencer.inference()
    prediction = data_pipeline.decode_categorical_features(prediction)
    save_submission(prediction, args, args.runname)

if __name__ == '__main__':
    main()
