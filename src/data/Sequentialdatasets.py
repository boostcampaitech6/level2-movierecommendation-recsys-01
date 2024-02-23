from collections import defaultdict
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from .datasets import DataPipeline

import logging
logger = logging.getLogger(__name__)


class SeqDataPipiline(DataPipeline):
    def __init__(self):
        super().__init__()

    def preprocess_data(self) -> dict:
        logger.info("[Seq] preprocess data...")
        data = self._read_data()
        data = self._sorted(data)
        # data = self._feature_engineering(data)
        data = self._feature_selection(data)
        data = self._data_formatting(data)
        return data
    
    def _sorted(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("[Seq] sorting data by user and time")
        data.sort_values(['user', 'time'], inplace=True)
        return data
    
    def _data_formatting(self, df) -> dict:
        logger.info("[Seq] data formatting to sequence per user dict...")
        df = df.groupby('user')['item'].agg(list).to_dict()
        return df
    
    def split_data(self, data: dict):
        logger.info("[Seq] split data...")

        # users_seq = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
        # user_train = {}
        # user_valid = {}
        # for user, item, time in zip(data['user'], data['item'], data['time']):
        #     users_seq[user].append(item)

        train_X, train_y = defaultdict(list), defaultdict(list)
        valid_X, valid_y = defaultdict(list), defaultdict(list)
        for user, seq in data.items():
            train_X[user] = seq[:-2] # train은 전체 sequence에서 마지막 2개 제외
            train_y[user] = seq[-2] # target을 마지막에서 2번째 아이템으로

            valid_X[user] = seq[:-1] # valid는 전체 sequence에서 마지막 1개 제외
            valid_y[user] = seq[-1] # target을 마지막 아이템으로
            
        train_data = {'X': train_X, 'y': train_y} # 
        valid_data = {'X': valid_X, 'y': valid_y} # 
        evaluate_data = data # 전체 유저별 시퀀스가 evaluate의 기본 데이터
        return train_data, valid_data, evaluate_data


class SeqDataset(Dataset):
    def __init__(self, data, max_len, mask_prob, is_train=False):
        super().__init__()
        self.X = data['X']
        if is_train:
            self.y = data['y']
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, user):
        user_seq = self.X[user] # 유저의 sequence
        items, y = self._cloze(user_seq)
        items = items[-self.max_len:]
        y = y[-self.max_len:]
        padding_len = self.max_len - len(items)

        # zero padding
        items = [0] * padding_len + items
        y = [0] * padding_len + y
        return {'X': torch.LongTensor(items), 'y': torch.LongTensor(y)}
    
    def _cloze(self, user_seq):
        items = []
        y = []
        for seq in user_seq:
            prob = np.random.random() # 마스킹 확률
            if prob < self.mask_prob:
                prob /= self.mask_prob # [0, 1] 로 정규화하기 위함

                # cloze 처리: 
                if prob < 0.8:
                    # masking
                    items.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    items.append(np.random.randint(1, self.num_item+1))  # item random sampling
                else:
                    items.append(seq)
                y.append(seq)  # 학습에 사용
            else:
                items.append(seq)
                y.append(0)  # 학습에 사용 X, trivial
        return items, y
