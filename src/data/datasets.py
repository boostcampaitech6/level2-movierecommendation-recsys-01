# datasets.py
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

from .features import (
    make_writer, make_director, make_genre, make_title, make_year
)

class DataPipeline:

    def __init__(self, args):
        self.args = args
        self.ordinal_encoder = None
        self.num_features = None
        self.cat_features_size = None

    def _read_data(self):
        logger.info('read data...')
        # read csv
        path = '/data/ephemeral/data/train/train_ratings.csv'
        df = pd.read_csv(path)
        return df

    def _neg_sampling(self, df):
        logger.info('negative sampling...')
        '''
        input: df (pos only)
        return: df (pos + neg)
        '''
        # positive rating
        df['rating'] = 1
        unique_items = df.item.unique()
        
        # positive 파악
        user_items = df.groupby('user').agg(pos_items=('item', set))

        # positive 빼기
        neg_items = user_items['pos_items'].apply(lambda x: set(unique_items)-x)

        # sampling
        user_items['neg_items'] = neg_items.apply(
                lambda x: np.random.choice(list(x), size=self.args.neg_count))
        
        # series to dataframe
        # user, item
        neg_samples_df = user_items.explode('neg_items')
        neg_samples_df = neg_samples_df.drop(columns=['pos_items']).reset_index()
        neg_samples_df = neg_samples_df.rename(columns={'neg_items': 'item'})
        neg_samples_df['rating'] = 0

        df = pd.concat([df, neg_samples_df], axis=0).reset_index(drop=True)

        return df
    
    def _feature_engineering(self, df):
        logger.info('feature engineering...')
        make_year(df)
        make_director(df)
        make_genre(df)
        make_title(df)
        make_writer(df)
        return df
    
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
        logger.info("preprocess data...")
        df = self._read_data()
        df = self._neg_sampling(df)
        df = self._feature_engineering(df)
        df = self._feature_selection(df)
        data = self._data_formatting(df)
        return data

    def save_data(self, data, data_name):
        with open(data_name, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, data_name):
        logger.info("load data files...")
        with open(data_name, 'rb') as f:
            data = pickle.load(f)
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

        return train_data, valid_data


class FMDataset(Dataset):
    def __init__(self, data, train=False):
        super().__init__()
        self.X = data['X'].values.astype(int)
        self.train = train
        if self.train:
            self.y = data['y'].values.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return {'X': self.X[index], 'y': self.y[index]}
