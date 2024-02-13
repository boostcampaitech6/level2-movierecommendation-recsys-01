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
import logging

@hydra.main(config_path="./src/configs", config_name="train_config", version_base='1.3')
def main(args: DictConfig):

    # runname
    time_zone = tz.timezone('Asia/Seoul')
    now = dt.strftime(dt.now(time_zone), '%y%m%d-%H%M%S')
    runname = f"{args.model_name}_{now}"
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.submit_dir).mkdir(exist_ok=True, parents=True)
    
    # 로그 파일 설정
    log_file = args.log_dir + runname + '.log'
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # 파일 핸들러를 생성합니다.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 파일 핸들러에 포맷을 설정합니다.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 파일 핸들러를 루트 로거에 추가합니다.
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logging.info(f"run name: {runname}")

    # wandb init
    if args.wandb:
        logging.info("wandb init...")
        wandb.init(project=args.project, config=dict(args), name=runname)

    # seed
    set_seed(args.seed)

    # create data_path
    data_path, train_path, valid_path, evaluate_path = create_data_path(args)
    data_pipeline = DataPipeline(args)

    if (not os.path.exists(train_path)) or (not os.path.exists(valid_path)) \
        or (not os.path.exists(evaluate_path)) or args.data_rebuild :
        logging.info("build datasets...")
        Path(data_path).mkdir(exist_ok=True, parents=True)
        data = data_pipeline.preprocess_data()
        train_data, valid_data, evaluate_data = data_pipeline.split_data(data)

        data_pipeline.save_data(train_data, train_path)
        data_pipeline.save_data(valid_data, valid_path)
        data_pipeline.save_data(evaluate_data, evaluate_path)
    else:
        logging.info("using saved datasets...")
        train_data = data_pipeline.load_data(train_path)
        valid_data = data_pipeline.load_data(valid_path)
        evaluate_data = data_pipeline.load_data(evaluate_path)

    # ordinal encoding
    logging.info("encode categorical features...")
    cat_features = [name for name, options in args.feature_sets.items() if options == [1, 'C']]
    train_data['X'] = data_pipeline.encode_categorical_features(train_data['X'], cat_features)
    valid_data['X'] = data_pipeline.encode_categorical_features(valid_data['X'], cat_features)
    evaluate_data = data_pipeline.encode_categorical_features(evaluate_data, cat_features)

    # scaling
    logging.info("scaling numeric features...")
    num_features = [name for name, options in args.feature_sets.items() if options == [1, 'N']]
    train_data['X'] = data_pipeline.scale_numeric_features(train_data['X'], num_features)
    valid_data['X'] = data_pipeline.scale_numeric_features(valid_data['X'], num_features)
    evaluate_data = data_pipeline.scale_numeric_features(evaluate_data, num_features)

    # dataset
    logging.info("make Dataset...")
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
    
    # dataloader
    logging.info("make DataLoader...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Trainer 
    logging.info("make Trainer...")
    trainer = Trainer(args, evaluate_data, data_pipeline, runname)
    trainer.run(train_dataloader, valid_dataloader)

    # Load Best Model
    logging.info("using saved datasets...")
    trainer.load_best_model()

    # Inference
    logging.info("using saved datasets...")
    prediction = trainer.inference()
    prediction = data_pipeline.decode_categorical_features(prediction)
    save_submission(prediction, args, runname)

    # wandb finish
    if args.wandb:
        logging.info("wandb finish...")
        wandb.finish()
    
if __name__ == '__main__':
    main()
