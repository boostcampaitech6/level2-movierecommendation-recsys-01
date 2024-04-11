from abc import *
from tqdm import tqdm

import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

from .datasets import DataPipeline
from . import features

class FMDataPipeline(DataPipeline):

    def __init__(self, args):
        super().__init__(args)
        self.ordinal_encoder = None
        self.num_features = None
        self.cat_features_size = None
        self.map_file_name = None
    
    def _feature_selection(self, df):
        logger.info('feature selection...')
        # ordering to con_features | cat_features
        num_features = [key for key, value in self.args.feature_sets.items() if value == [1, 'N']]
        cat_features = [key for key, value in self.args.feature_sets.items() if value == [1, 'C']]
        df = df[num_features + cat_features]
        return df

    def _data_formatting(self, df: pd.DataFrame):
        logger.info('data formatting...')
        return {
            'X': df.drop(['negative', 'positive_y', 'negative_y'], axis=1).rename(columns={'positive': 'item'}),
            'positive_y': df[['positive_y']],
            'y': df.drop(['positive', 'positive_y', 'negative_y'], axis=1).rename(columns={'negative': 'item'}),
            'negative_y': df[['negative_y']],
        }
    
    def _neg_sampling(self, df):
        logger.info('negative sampling...')
        '''
        input: df (pos only)
        return: df (pos + neg)
        '''

        # unique items set
        unique_items = df.item.unique()

        # negative items set
        user_items = df.groupby('user').agg(positive=('item', set)).to_dict()['positive']

        # (user, positive, negative) 형태의 데이터프레임 생성
        result = []
        for user, positive in tqdm(user_items.items()):
            negative = list(set(unique_items) - positive)
            # negative 항목 중 200개만 샘플링
            for pos in positive:
                sampled_negative = random.sample(negative, min(self.args.neg_count, len(negative)))
                result.extend([(user, pos, 1, neg, 0) for neg in sampled_negative])
        
        logger.info(f"[FM] neg sampling end... {len(result)}")
        result_df = pd.DataFrame(result, columns=['user', 'positive', 'positive_y', 'negative', 'negative_y'])
        logger.info("[FM] result dataframe create done...")
        logger.info(result_df.sample(10))

        return result_df

    def preprocess_data(self):
        # abstract
        logger.info("preprocess data...")
        df = self._read_data()
        df = self._neg_sampling(df)
        data = self._data_formatting(df)
        data['X'], data['y'] = self._feature_engineering(None, data['X']), self._feature_engineering(data['X'], data['y'])
        data['X'], data['y'] = self._feature_selection(data['X']), self._feature_selection(data['y'])

        # save positive interaction mapping data
        logger.info("[FM]save feature map data...")
        feature_bit = ''.join([value[1] if value[0] == 1 else '0' for value in self.args.feature_sets.values()])
        # Remove trailing zeros from the feature bit string to ensure consistent feature set recognition
        # regardless of the feature list size. This normalization step prevents different interpretations
        # of the same feature set due to varying lengths of trailing zeros.
        feature_bit = feature_bit.rstrip('0')
        data_path = f'{self.args.data_dir}/{self.args.model_name}-{self.args.neg_count}-{feature_bit}'
        file_name = f'{data_path}/features.parquet'
        self.map_file_name = file_name
        self.save_data_parquet(data['X'], self.map_file_name)

        return data

    def encode_categorical_features(self, df, cat_features):
        if self.ordinal_encoder is None: # train-only
            logger.info("[Train] make ordinal encoder and fit...")
            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_encoder = self.ordinal_encoder.fit(df[cat_features])
            self.cat_features = cat_features

        logger.info("transform features...")
        df[cat_features] = self.ordinal_encoder.transform(df[cat_features]).astype(int)

        self.cat_features_size = {cat_name: len(categories) for cat_name, categories in zip(
            cat_features, self.ordinal_encoder.categories_)}

        return df

    def decode_categorical_features(self, array):
        return self.ordinal_encoder.inverse_transform(array)
    
    def scale_numeric_features(self, df, num_features):
        logger.info("scaling features...")
        self.num_features = num_features
        if len(num_features) == 0:
            logger.info("no numeric features...")
            return df
        
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features])

        return df

    def split_data(self, data):
        logger.info('split data...')
        # split by user and y
        X_train, X_valid, y_train, y_valid = [],[],[],[]
        pos_y_train, pos_y_valid, neg_y_train, neg_y_valid = [], [], [], []

        for _, user_data in tqdm(data['X'].groupby('user')):
            user_negative = data['y'].iloc[user_data.index]
            user_positive_y = data['positive_y'].iloc[user_data.index]
            user_negative_y = data['negative_y'].iloc[user_data.index]

            (user_data_train, user_data_valid,
             user_positive_y_train, user_positive_y_valid,
             user_negative_train, user_negative_valid,
             user_negative_y_train, user_negative_y_valid,
             ) = train_test_split(user_data, user_positive_y, user_negative, user_negative_y, test_size=.2)

            X_train.append(user_data_train) # positive data
            X_valid.append(user_data_valid)
            y_train.append(user_negative_train) # negative data
            y_valid.append(user_negative_valid)
            pos_y_train.append(user_positive_y_train) # 
            pos_y_valid.append(user_positive_y_valid)
            neg_y_train.append(user_negative_y_train) # 
            neg_y_valid.append(user_negative_y_valid)
        
        # concat
        X_train, X_valid = pd.concat(X_train), pd.concat(X_valid)
        y_train, y_valid = pd.concat(y_train), pd.concat(y_valid)
        positive_y_train, positive_y_valid = pd.concat(pos_y_train), pd.concat(pos_y_valid)
        negative_y_train, negative_y_valid = pd.concat(neg_y_train), pd.concat(neg_y_valid)

        train_data = {'X': X_train, 'y': y_train, 'positive_y': positive_y_train, 'negative_y': negative_y_train}
        valid_data = {'X': X_valid, 'y': y_valid, 'positive_y': positive_y_valid, 'negative_y': negative_y_valid}
        evaluate_data = self._input_of_total_user_item() # array -> trainer 

        return train_data, valid_data, evaluate_data

    def _input_of_total_user_item(self):
        num_users = len(self.users)
        num_items = len(self.items)
        
        # Create user-item interaction matrix
        logger.info("Make base users and items interaction....")

        users = self.users[:, None]
        items = self.items[None, :]
        data = np.column_stack(
            (np.repeat(users, num_items, axis=1).flatten(), np.tile(items, (num_users, 1)).flatten())
        )

        # Convert to DataFrame
        data = pd.DataFrame(data, columns=['user', 'item'])

        assert len(data) == (num_users * num_items), f"Total Interaction이 부족합니다: {len(data)}"

        # mapping side informations - 6 minutes...
        logger.info("Map side informations...")
        map_df = self.load_data_parquet(self.map_file_name)
        self._feature_engineering(map_df, data)

        # ordering
        logger.info("Ordering numeric and categorical features...")
        num_features = [name for name, options in self.args.feature_sets.items() if options == [1, 'N']]
        cat_features = [name for name, options in self.args.feature_sets.items() if options == [1, 'C']]
        # data = pd.concat([data[num_features], data[cat_features]], axis=1)
        logger.info(data.memory_usage(deep=True))
        logger.info(data.dtypes)
        data = data[num_features + cat_features]
        # data = pd.concat([data.loc[:, num_features], data.loc[:, cat_features]], axis=1)

        return data
    
    def _feature_engineering(self, origin, target):
        logger.info('[FM]feature engineering...')
        features_attributes = dir(features)
        use_features = [key for key, value in self.args.feature_sets.items() if value[0] == 1]
        for feature in use_features:
            if feature in ('user', 'item', 'time'): # skip for user and item and time
                continue
            if feature not in features_attributes:
                logger.error(f"[ERROR] '{feature}' make method is not in features.py module...")
                raise ValueError
            make_feature = getattr(features, feature)
            make_feature(origin, target)
        return target


class FMDataset(Dataset):
    def __init__(self, data, train=False):
        super().__init__()
        self.X = data['X'].values.astype(np.float32)
        self.y = data['y'].values.astype(np.float32)
        self.train = train
        if self.train:
            self.positive_y = data['positive_y'].values.astype(np.float32)
            self.negative_y = data['negative_y'].values.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return {
            'positive_X': self.X[index], # (batch_size, user, item)
            'positive_y': self.positive_y[index], # (batch_size, 1)
            'negative_X': self.y[index], # (batch_size, user, item)
            'negative_y': self.negative_y[index], # (batch_size, 0)
        }
