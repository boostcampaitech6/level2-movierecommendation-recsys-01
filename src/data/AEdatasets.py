from abc import *

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

from .datasets import DataPipeline


class AEDataPipeline(DataPipeline):

    def __init__(self, args):
        super().__init__(args)
        self.users = None
        self.items = None

    def _feature_selection(self, df):
        logger.info('feature selection...')
        df = df[['user', 'item', 'rating']]
        return df

    def _concat_neg_data(self, df):
        logger.info('concat all negative data...')
        users = df.user.unique()
        items = set(df.item.unique())
        # positive rating
        df['rating'] = 1
        neg_data = []
        pos_items = df.groupby('user')['item'].agg(set)
        for user in tqdm(users):
            for item in (items-pos_items[user]):
                neg_data.append([user,item,0,0])
        df = pd.concat([df, pd.DataFrame(neg_data, columns=df.columns)])
        return df

    def preprocess_data(self):
        logger.info("preprocess data...")
        df = self._read_data()
        df = self._concat_neg_data(df)
        df = self._feature_selection(df)
        return df 
    
    def _transform_df_to_uimatrix(self, df):
        uimatrix = df.pivot_table(index='user', columns=['item'], values=['rating'], aggfunc='first')
        uimatrix.columns = uimatrix.columns.droplevel(0)
        uimatrix = uimatrix.reindex(index=self.users, columns=self.items, fill_value=np.nan)
        return uimatrix

    def _make_mask(self, interact):
        logger.info('make mask...')
        all_mask = interact.notnull() # pos+neg
        pos_mask = interact == 1 # pos only
        return all_mask, pos_mask

#    def efficient_split_data(self, df):
#        # not completed
#        users = df.user.unique()
#        items = set(df.item.unique())
#        # positive rating
#        df['rating'] = 1
#        neg_data = []
#        pos_items = df.groupby('user')['item'].agg(set)
#        split_ratio = .2
#        for user in tqdm(users):
#
#            for item in (items-pos_items[user]):
#                neg_data.append([user,item,0,0])
#        df = pd.concat([df, pd.DataFrame(neg_data, columns=df.columns)])
#        return df


    def split_data(self, df):
        logger.info('split data...')
        # drop duplicates
        df = df.drop_duplicates(subset=['user','item'])
        # split by user and y
        X_train, X_valid, y_train, y_valid = [],[],[],[]

        for _, user_data in tqdm(df.groupby('user')):
            user_y = user_data['rating']

            user_data_train, user_data_valid, user_y_train, user_y_valid = \
                    train_test_split(user_data, user_y, test_size=.2,
                    stratify=user_y)

            X_train.append(user_data_train)
            X_valid.append(user_data_valid)
        
        # concat
        X_train, X_valid = pd.concat(X_train), pd.concat(X_valid)
        train_df = pd.DataFrame(X_train, columns=['user','item','rating'])
        valid_df = pd.DataFrame(X_valid, columns=['user','item','rating'])

        # transform df to uimatrix
        train_interact = self._transform_df_to_uimatrix(train_df) # pos 1, neg 0
        valid_interact = self._transform_df_to_uimatrix(valid_df)
        full_interact = self._transform_df_to_uimatrix(df)

        # get mask
        train_interact_all_mask, train_interact_pos_mask = self._make_mask(train_interact) # pos 1, neg 0
        valid_interact_all_mask, valid_interact_pos_mask = self._make_mask(valid_interact) # pos 1, neg 0
        full_interact_all_mask, full_interact_pos_mask = self._make_mask(full_interact) # pos 1, neg 0

        # get positive items
        train_pos_items = train_df[train_df['rating']==1].groupby('user')['item'].agg(set).sort_index().to_dict()
        valid_pos_items = valid_df[valid_df['rating']==1].groupby('user')['item'].agg(set).sort_index().to_dict()
        full_pos_items = df[df['rating']==1].groupby('user')['item'].agg(set).sort_index().to_dict()

        train_data = {'interact': full_interact, 'interact_all_mask': train_interact_all_mask,
                      'interact_pos_mask': train_interact_pos_mask, 'pos_items': train_pos_items}
        valid_data = {'interact': full_interact, 'interact_all_mask': valid_interact_all_mask,
                      'interact_pos_mask': valid_interact_pos_mask, 'pos_items': valid_pos_items}
        full_data = {'interact': full_interact, 'interact_all_mask': full_interact_all_mask,
                      'interact_pos_mask': full_interact_pos_mask, 'pos_items': full_pos_items}
        
        return train_data, valid_data, full_data

    def postprocessing(self, data):
        data['interact'] = data['interact'].fillna(0.5).values.astype(np.float32)
        data['interact_all_mask'] = data['interact_all_mask'].values # compare (pred * mask) with X
        data['interact_pos_mask'] = data['interact_pos_mask'].values # compare (pred * mask) with X
        return data

    def set_unique_users_items(self, train_data):
        self.users = train_data['interact'].index.to_numpy()
        self.items = train_data['interact'].columns.to_numpy()

class AEDataset(Dataset):

    def __init__(self, data):
        '''[1,0,0,1,0,1]'''
        super().__init__()
        self.interact = data['interact']
        self.interact_all_mask = data['interact_all_mask']
        self.interact_pos_mask = data['interact_pos_mask']

    def __len__(self):
        return self.interact.shape[0]

    def __getitem__(self, index):
        return {'interact': self.interact[index], 
                'interact_all_mask': self.interact_all_mask[index], 
                'interact_pos_mask': self.interact_pos_mask[index]}
