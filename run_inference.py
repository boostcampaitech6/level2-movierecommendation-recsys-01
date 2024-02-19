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

from src.data.FMdatasets import FMDataPipeline, FMDataset
from src.data.AEdatasets import AEDataPipeline, AEDataset
from src.inference.FMinferencer import FMInferencer
from src.inference.AEinferencer import AEInferencer

#from src.data.datasets import (
#    DataPipeline,
#    )
#from src.data.FMdatasets import (
#    FMDataPipeline, FMDataset,
#    ) 
#from src.inference import Inferencer
from src.utils import set_seed, create_data_path, save_submission

import torch
from torch.utils.data import DataLoader

@hydra.main(config_path="./src/configs", config_name="inference_config", version_base='1.3')
def main(args: DictConfig):

    # create data_path
    data_path, train_path, valid_path, evaluate_path = create_data_path(args)
    if args.model_name in ('FM', 'DeepFM'):
        data_pipeline = FMDataPipeline(args)
    elif args.model_name.endswith('AE'):
        data_pipeline = AEDataPipeline(args)
    else:
        raise ValueError()

    print("using saved datasets...")
    train_data = data_pipeline.load_data(train_path)
    valid_data = data_pipeline.load_data(valid_path)
    evaluate_data = data_pipeline.load_data(evaluate_path)

    # set unique users and items
    data_pipeline.set_unique_users_items(train_data)

    # postprocessing
    print("post-processing...")
    train_data = data_pipeline.postprocessing(train_data)
    valid_data = data_pipeline.postprocessing(valid_data)
    evaluate_data = data_pipeline.postprocessing(evaluate_data)

    # Load Best Model
    inferencer = AEInferencer(args, evaluate_data, data_pipeline, args.runname)

    # Inference
    prediction = inferencer.inference(evaluate_data)
    save_submission(prediction[:, :2], args, args.runname)

if __name__ == '__main__':
    main()
