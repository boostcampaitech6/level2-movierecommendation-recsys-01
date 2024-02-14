import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from .models.DeepFMModels import DeepFM
from .models.FMModels import FM

class Inferencer:

    def __init__(self, args, data_pipeline, train_dataloader, valid_dataloader, evaluate_data):

        self.args = args
        self.device = torch.device("cuda")
        self.num_features = data_pipeline.num_features
        self.cat_features_size = data_pipeline.cat_features_size

        self.model = None
        self.train_actual = None
        self.valid_actual = None

        self._load_model()

        self.train_actual = None
        self.valid_actual = None
        self._set_actuals(train_dataloader, valid_dataloader)

        self.total_interaction = torch.tensor(evaluate_data.values).to(self.device)

    def _actual_interaction_dict(self, X):
        offset = len(self.num_features)
        return pd.DataFrame(X[:, offset:offset+2], columns = ['user', 'item']).groupby('user')['item'].agg(set).sort_index().to_dict()

    def _set_actuals(self, train_dataloader, valid_dataloader):
        print('set actuals...')

        train_pos_X, valid_pos_X = [], []
        for data in train_dataloader:
            positive_index = torch.where(data['y'][:,0] == 1)
            train_pos_X.append(data['X'][positive_index])

        for data in train_dataloader:
            positive_index = torch.where(data['y'][:,0] == 1)
            valid_pos_X.append(data['X'][positive_index])

        train_pos_X = np.concatenate(train_pos_X, axis=0)
        valid_pos_X = np.concatenate(valid_pos_X, axis=0)

        self.train_actual = self._actual_interaction_dict(train_pos_X)
        self.valid_actual = self._actual_interaction_dict(valid_pos_X)

    def _load_model(self):
        print('load best model...')
        if self.args.model_name == "FM":
            self.model = FM(self.num_features, self.cat_features_size, self.args.emb_dim)
        elif self.args.model_name == "DeepFM":
            self.model = DeepFM(self.num_features, self.cat_features_size, self.args.emb_dim, mlp_dims=[200, 200, 200], drop_rate=0.1)
        else:
            raise Exception

        self.model.to(self.device)
        self.model.eval()

        print(f'load {self.args.runname} model...')
        self.model.load_state_dict(torch.load(f'{self.args.model_dir}/{self.args.runname}/best_model.pt'))

        print('best model info...')
        with open(f'{self.args.model_dir}/{self.args.runname}/best_model_info.txt', 'r') as f:
            info = f.readlines()
        print(''.join(info))

    def inference(self, k=10):
        print("Inference Start....")
        self.model.eval()

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        offset = len(self.num_features)
        prediction = []

        print("Predict all users and items interaction....")
        users = self.total_interaction[:, offset].unique().detach().cpu().numpy()
        for idx, user in enumerate(tqdm(users)):
            start_idx, end_idx = idx * num_items, (idx+1) * num_items
            user_X = self.total_interaction[start_idx:end_idx, :]
            user_items = user_X.detach().cpu().numpy()[:, offset+1]
            user_mask = torch.tensor([0 if (
                item.item() in self.train_actual[int(user)]) or (item.item() in self.valid_actual[int(user)]) else 1 for item in user_items], dtype=int)
            
            user_pred = self.model(user_X.float()).detach().cpu()
            user_pred = user_pred.squeeze(1) * user_mask # train interaction 제외
            
            # find high prob index
            high_index = np.argpartition(user_pred.numpy(), -k)[-k:]
            # find high prob item by index
            user_recom = user_items[high_index]
            prediction.append(user_recom)
        
        # expand_dims
        prediction = np.expand_dims(np.concatenate(prediction, axis=0), axis=-1)
        user_ids = np.expand_dims(np.repeat(users, 10), axis=-1).astype(int)

        prediction = np.concatenate([user_ids, prediction], axis=1)

        return prediction
