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
    DataPipeline,
    )
from src.data.FMdatasets import (
    FMDataPipeline, FMDataset,
    ) 
from src.inference import Inferencer
from src.utils import set_seed, create_data_path, save_submission

import torch
from torch.utils.data import DataLoader

@hydra.main(config_path="./src/configs", config_name="inference_config", version_base='1.3')
def main(args: DictConfig):
    # create data_path
    data_path, train_path, valid_path, evaluate_path = create_data_path(args)
    if args.model_name in ('FM', 'DeepFM'):
        data_pipeline = FMDataPipeline(args)
    else:
        raise ValueError()

    # 데이터 이미 있다고 가정
    train_data = data_pipeline.load_data(train_path)
    valid_data = data_pipeline.load_data(valid_path)
    evaluate_data = data_pipeline.load_data(evaluate_path)

    # ordinal encoding
    print("encode categorical features...")
    cat_features = [name for name, options in args.feature_sets.items() if options == [1, 'C']]
    train_data['X'] = data_pipeline.encode_categorical_features(train_data['X'], cat_features)
    valid_data['X'] = data_pipeline.encode_categorical_features(valid_data['X'], cat_features)
    evaluate_data = data_pipeline.encode_categorical_features(evaluate_data, cat_features)

    # scaling
    print("scaling numeric features...")
    num_features = [name for name, options in args.feature_sets.items() if options == [1, 'N']]
    train_data['X'] = data_pipeline.scale_numeric_features(train_data['X'], num_features)
    valid_data['X'] = data_pipeline.scale_numeric_features(valid_data['X'], num_features)
    evaluate_data = data_pipeline.scale_numeric_features(evaluate_data, num_features)
    
    # dataset
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2048, shuffle=True)

    # Load Best Model
    inferencer = Inferencer(args, data_pipeline, train_dataloader, valid_dataloader, evaluate_data)

    # Inference
    prediction = inferencer.inference()
    prediction = data_pipeline.decode_categorical_features(prediction)
    save_submission(prediction, args, args.runname)

if __name__ == '__main__':
    main()
