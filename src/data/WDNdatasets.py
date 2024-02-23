from abc import *
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

from .datasets import DataPipeline
from . import features

class WDNDataPipeline(DataPipeline):

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
        rating = ['rating']
        df = df[num_features + cat_features + rating]
        return df

    def _data_formatting(self, df):
        logger.info('data formatting...')
        return {
            'X': df.drop('rating', axis=1),
            'y': df[['rating']],
        }

    def preprocess_data(self):
        # abstract
        logger.info("preprocess data...")
        df = self._read_data()
        df = self._neg_sampling(df)
        print(df[df.user==11])
        # df = self._feature_engineering(None, df)
        df = self._feature_selection(df)
        data = self._data_formatting(df)

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

        for _, user_data in tqdm(data['X'].groupby('user')):
            user_y = data['y'].iloc[user_data.index]

            user_data_train, user_data_valid, user_y_train, user_y_valid = \
                    train_test_split(user_data, user_y, test_size=.2,
                    stratify=user_y)

            X_train.append(user_data_train)
            X_valid.append(user_data_valid)
            y_train.append(user_y_train)
            y_valid.append(user_y_valid)
        
        # concat
        X_train, X_valid = pd.concat(X_train), pd.concat(X_valid)
        y_train, y_valid = pd.concat(y_train), pd.concat(y_valid)

        train_data = {'X': X_train, 'y': y_train}
        valid_data = {'X': X_valid, 'y': y_valid}
        evaluate_data = self._input_of_total_user_item() # array -> trainer 

        return train_data, valid_data, evaluate_data

    def _input_of_total_user_item(self):
        num_items = len(self.items)
        
        # Create user-item interaction matrix
        logger.info("Make base users and items interaction....")
        num_samples = self.args.evaluate_size
        num_rows = self.users.shape[0]

        # 랜덤한 인덱스 선택
        random_indices = np.random.choice(num_rows, size=num_samples, replace=False)

        # 랜덤하게 선택된 샘플 추출
        users = self.users[random_indices, None]
        items = self.items[None, :]
        data = np.column_stack(
            (np.repeat(users, num_items, axis=1).flatten(), np.tile(items, (num_samples, 1)).flatten())
        )

        # Convert to DataFrame
        data = pd.DataFrame(data, columns=['user', 'item'])

        assert len(data) == (num_samples * num_items), f"Total Interaction이 부족합니다: {len(data)}"

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


class WDNDataset(Dataset):
    def __init__(self, data, train=False):
        super().__init__()
        self.X = data['X'].values.astype(np.float32)
        self.train = train
        if self.train:
            self.y = data['y'].values.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return {'X': self.X[index], 'y': self.y[index]}
