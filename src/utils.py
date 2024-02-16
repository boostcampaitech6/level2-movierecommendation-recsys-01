# utils.py
import os
import random

import numpy as np
import pandas as pd

import torch

import logging
logger = logging.getLogger(__name__)


def set_seed(seed):
    logger.info("seed setting...")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def create_data_path(args):
    logger.info("create data path...")
    # data_path
    feature_bit = ''.join([value[1] if value[0] == 1 else '0' for value in args.feature_sets.values()])
    # Remove trailing zeros from the feature bit string to ensure consistent feature set recognition
    # regardless of the feature list size. This normalization step prevents different interpretations
    # of the same feature set due to varying lengths of trailing zeros.
    feature_bit = feature_bit.rstrip('0')

    data_path = f'{args.data_dir}/{args.model_name}-{args.neg_count}-{feature_bit}'
    train_data_path = f'{data_path}/train.pickle'
    valid_data_path = f'{data_path}/valid.pickle'
    if args.model_name in ('FM', 'DeepFM'):
        evaluate_data_path = f'{data_path}/evaluate.parquet'
    else:
        evaluate_data_path = f'{data_path}/evaluate.pickle'

    return data_path, train_data_path, valid_data_path, evaluate_data_path


def save_submission(prediction, args, runname):
    logger.info("save submission file...")
    submission_df = pd.read_csv('../data/eval/sample_submission.csv')
    submission_df.iloc[:,:] = prediction
    submission_df.to_csv(f'{args.submit_dir}/{runname}-submission.csv', index=False)

