import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from .models.DeepFMModels import DeepFM
from .models.FMModels import FM

class Inferencer:

    def __init__(self, args, cat_features_size, train_dataloader, valid_dataloader):

        self.args = args
        self.device = torch.device("cuda")
        self.cat_features_size = cat_features_size

        self.model = None
        self.train_actual = None
        self.valid_actual = None

        self._load_model()
        self._set_actuals(train_dataloader, valid_dataloader)

    def _actual_interaction_dict(self, X):
        return pd.DataFrame(X, columns = ['user', 'item']).groupby('user')['item'].agg(set).sort_index().to_dict()

    def _set_actuals(self, train_dataloader, valid_dataloader):
        print('set actuals...')

        self.model.eval()
        train_X = np.concatenate([data['X'] for data in train_dataloader], axis=0)
        valid_X = np.concatenate([data['X'] for data in valid_dataloader], axis=0)

        self.train_actual = self._actual_interaction_dict(train_X)
        self.valid_actual = self._actual_interaction_dict(valid_X)

    def _load_model(self):
        print('load model...')
        if self.args.model_name == "FM":
            self.model = FM(self.cat_features_size, self.args.emb_dim)
        elif self.args.model_name == "DeepFM":
            self.model = DeepFM(self.cat_features_size, self.args.emb_dim, mlp_dims=[200, 200, 200], drop_rate=0.1)
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

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        prediction = []

        print("Predict all users and items interaction....")
        for _, user in enumerate(tqdm(range(num_users))):
            user_X = torch.tensor([[user, item] for item in range(num_items)], dtype=int).to(self.device)
            user_mask = torch.tensor([0 if (
                item in self.train_actual[user]) or (item in self.valid_actual[user]) else 1 for item in range(num_items)], dtype=int)
            
            user_pred = self.model(user_X).detach().cpu()
            user_pred = user_pred.squeeze(1) * user_mask # train interaction 제외
            
            user_pred = np.argpartition(user_pred.numpy(), -k)[-k:]
            prediction.append(user_pred)
        
        # expand_dims
        prediction = np.expand_dims(np.concatenate(prediction, axis=0), axis=-1)
        user_ids = np.expand_dims(np.repeat(np.arange(num_users), 10), axis=-1)

        prediction = np.concatenate([user_ids, prediction], axis=1)

        return prediction
