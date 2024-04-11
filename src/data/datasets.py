# datasets.py
import pickle
from tqdm import tqdm
from abc import *

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import logging
logger = logging.getLogger(__name__)

from . import features

class DataPipeline:

    def __init__(self, args):
        self.args = args
        self.ordinal_encoder = None
        self.num_features = None
        self.cat_features_size = None
        self.users = None
        self.items = None

    def _read_data(self):
        logger.info('read data...')
        # read csv
        path = '/data/ephemeral/data/train/train_ratings.csv'
        df = pd.read_csv(path)
        self.users = df['user'].unique()
        self.items = df['item'].unique()
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

        df = pd.concat([df, neg_samples_df.astype(int)], axis=0).reset_index(drop=True)

        return df
    
    def _feature_engineering(self, df):
        logger.info('feature engineering...')
        features_attributes = dir(features)
        use_features = [key for key, value in self.args.feature_sets.items() if value[0] == 1]
        for feature in use_features:
            if feature in ('user', 'item', 'time'): # skip for user and item and time
                continue
            if feature not in features_attributes:
                logger.error(f"[ERROR] '{feature}' make method is not in features.py module...")
                raise ValueError
            make_feature = getattr(features, feature)
            make_feature(df)
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

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def split_data(self, data):
        pass

    def save_data(self, data, data_name):
        logger.info(f"saving {data_name}")
        with open(data_name, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, data_name):
        logger.info("load data files...")
        with open(data_name, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def save_data_parquet(self, data: pd.DataFrame, data_name):
        logger.info(f"saving {data_name}")
        data.to_parquet(data_name)

    def load_data_parquet(self, data_name):
        logger.info("load evaluate file...")
        return pd.read_parquet(data_name)

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
