from abc import *
from copy import deepcopy

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
        self.confidence_array = None
        if self.args.confidence is not None:
            self._set_confidence_array()

    def _set_confidence_array(self):
        logger.info('set confidence array...')
        df = self._read_data()
        if self.args.confidence == 'None':
            pass
        elif self.args.confidence == 'time':
            # calculate min, max of time
            user_time_min = df.groupby('user')['time'].min().to_dict()
            user_time_max = df.groupby('user')['time'].max().to_dict()
            # min-max normalization
            df['norm_time'] = df[['user', 'time']].apply(
                lambda x: (x['time']-user_time_min[x['user']])/(user_time_max[x['user']]-user_time_min[x['user']]), axis=1)
            confidence_matrix = df.pivot('user',['item'], ['norm_time']).fillna(0).astype('float32') + 1.
            confidence_matrix.columns = confidence_matrix.columns.droplevel(0)
            self.confidence_array = confidence_matrix.loc[self.users,self.items].values

        elif self.args.confidence == 'itempop':
            popularity = df.value_counts('item')
            df = df.merge(pd.DataFrame(popularity, columns=['itempop']), on='item')
            df['itempop'] = (df['itempop']-df["itempop"].min())/(df["itempop"].max()-df["itempop"].min())
            confidence_matrix = df.pivot('user',['item'], ['itempop']).fillna(0) + 1.
            confidence_matrix.columns = confidence_matrix.columns.droplevel(0)
            self.confidence_array = confidence_matrix.loc[self.users,self.items].values
        elif self.args.confidence == 'genreprefer':

            # read genre data
            genre_data = pd.read_csv('../data/train/genres.tsv', sep='\t')
            # genre pivotting
            genre_data['rating'] = 1
            genre_df = genre_data.pivot(index='item', columns='genre', values='rating').fillna(0)
            # genre merging
            genre_merged_df = df.merge(genre_df, on='item')
            # genre count by user
            user_genre_matrix = genre_merged_df.groupby('user').sum().iloc[:,3:]
            # genre scaling by user (-> preference)
            user_genre_matrix = user_genre_matrix.div(user_genre_matrix.sum(axis=1), axis=0)

            # 장르 컬럼 리스트 (예시 데이터에 맞게 조정)
            genre_columns = [col for col in user_genre_matrix.columns if col not in ['user', 'item', 'time', 'itempop']]

            # 1단계: 사용자별 장르 선호도를 genre_merged_df에 결합
            # user_genre_df의 인덱스가 user_id라고 가정하고, reset_index()를 호출하여 'user' 컬럼을 생성
            user_genre_df_reset = user_genre_matrix.reset_index()

            # genre_merged_df와 user_genre_df를 'user' 컬럼을 기준으로 결합
            merged_df = pd.merge(genre_merged_df, user_genre_df_reset, on='user', how='left', suffixes=('', '_pref'))

            # 2단계: 벡터화된 연산을 사용하여 각 아이템에 대한 사용자의 장르 선호도 계산
            # 장르별 선호도와 아이템의 장르 정보를 곱하여 새로운 'item_genre_prefer' 컬럼 생성
            for genre in genre_columns:
                merged_df[genre + '_prefer'] = merged_df[genre] * merged_df[genre + '_pref']

            # 선택적: 장르 선호도의 합을 계산하여 각 아이템에 대한 총 선호도를 나타내는 컬럼 추가
            merged_df['total_genre_prefer'] = merged_df[[genre + '_prefer' for genre in genre_columns]].sum(axis=1)

            confidence_matrix = merged_df.pivot('user',['item'], ['total_genre_prefer']).fillna(0) + 1.
            confidence_matrix.columns = confidence_matrix.columns.droplevel(0)
            self.confidence_array = confidence_matrix.loc[self.users,self.items].values


        else:
            raise ValueError(f'{self.args.confidence} not found!')
        
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
        # self._set_confidence_array(df)
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
                      #'confidence': self.confidence_array}
        valid_data = {'interact': full_interact, 'interact_all_mask': valid_interact_all_mask,
                      'interact_pos_mask': valid_interact_pos_mask, 'pos_items': valid_pos_items}
                      #'confidence': self.confidence_array}
        full_data = {'interact': full_interact, 'interact_all_mask': full_interact_all_mask,
                      'interact_pos_mask': full_interact_pos_mask, 'pos_items': full_pos_items}

        return train_data, valid_data, full_data

    def postprocessing(self, data):
        data['interact'] = data['interact'].fillna(0.5).values.astype(np.float32)
        data['interact_all_mask'] = data['interact_all_mask'].values # compare (pred * mask) with X
        data['interact_pos_mask'] = data['interact_pos_mask'].values # compare (pred * mask) with X
        # data['confidence'] = data['confidence'].values # compare (pred * mask) with X
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
        #self.confidence = data['confidence']

    def __len__(self):
        return self.interact.shape[0]

    def __getitem__(self, index):
        return {'interact': self.interact[index], 
                'interact_all_mask': self.interact_all_mask[index], 
                'interact_pos_mask': self.interact_pos_mask[index]}
                #'confidence': self.confidence[index]}
