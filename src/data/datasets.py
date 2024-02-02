# datasets.py
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

def neg_sampling(train_df, num_users, num_items, num_negs):
    
    # positive 파악
    user_items = train_df.groupby('user').agg(pos_items=('item', set))

    # positive 빼기
    neg_items = user_items['pos_items'].apply(lambda x: set(range(num_items))-x)

    # sampling
    print('negative sampling...')
    user_items['neg_items'] = neg_items.apply(
            lambda x: np.random.choice(list(x), size=num_negs))
    print('done')
    
    # series to dataframe
    # user, item
    neg_samples_df = user_items.explode('neg_items')
    neg_samples_df = neg_samples_df.drop(columns=['pos_items']).reset_index()
    neg_samples_df = neg_samples_df.rename(columns={'neg_items': 'item'})
    neg_samples_df['rating'] = 0

    return neg_samples_df

def split_data(data):

    # split by user and y
    X_train, X_valid, y_train, y_valid = [],[],[],[]
    for user in tqdm(range(data['cat_features_size']['user'])):
        user_data = data['X'][data['X']['user'] == user]
        user_y = data['y'].iloc[user_data.index,:]
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

    train_data = {
        'X': X_train,
        'y': y_train,
        'cat_features_size': data['cat_features_size'], # dict
    }

    valid_data = {
        'X': X_valid,
        'y': y_valid,
        'cat_features_size': data['cat_features_size'], # dict
    }

    return train_data, valid_data

def save_data(data, data_name):
    with open(data_name, 'wb') as f:
        pickle.dump(data, f)

def load_data(data_name):
    with open(data_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_data():
    # read csv
    train_path = '/data/ephemeral/data/train/train_ratings.csv'
    train_df = pd.read_csv(train_path)

    # drop time
    train_df = train_df.drop(columns=['time'])

    # num users and items
    num_users = train_df.user.nunique()
    num_items = train_df.item.nunique()

    # ordinal encoding
    cat_features = ['user', 'item']
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder = ordinal_encoder.fit(train_df[cat_features])
    train_df[cat_features] = ordinal_encoder.transform(train_df[cat_features])
    cat_features_size = {cat_name: len(categories) for cat_name, categories in zip(
        cat_features, ordinal_encoder.categories_)}

    # astype
    train_df[cat_features] = train_df[cat_features].astype(int)

    # rating
    train_df['rating'] = 1

    # neg sample
    num_negs = 50
    neg_samples_df = neg_sampling(train_df, num_users, num_items, num_negs)
    train_df = pd.concat([train_df, neg_samples_df], axis=0)

    data = {
        'X': train_df.drop('rating', axis=1), # df
        'y': train_df[['rating']], # df
        'cat_features_size': cat_features_size, # dict
    }

    return data

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
