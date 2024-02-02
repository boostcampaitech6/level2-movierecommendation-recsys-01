'''
trainer.py

- run (train and valid)
- evaluate (metric)
- train (only loss)
- valid (metric)
'''
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
from .models.DeepFMModels import DeepFM
from .metrics import recall_at_k, ndcg_k

class Trainer():
    def __init__(self, args, cat_features_size) -> None:
        self.device = torch.device("cuda")
        self.args = args
        if self.args.model_name == "FM":
            self.model = FM(cat_features_size, self.args.emb_dim)
        elif self.args.model_name == "DeepFM":
            self.model = DeepFM(cat_features_size, self.args.emb_dim, mlp_dims=[200, 200, 200], drop_rate=0.1)
        else:
            raise Exception
        self.model.to(self.device)
        self.loss = torch.nn.BCELoss()
        self.cat_features_size = cat_features_size
      
    def get_model(self):
        return self.model
    
       
    def run(self, train_data_loader, valid_data_loader):
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.lr)
        else:
            raise Exception
        for epoch in range(self.args.epochs):
            train_loss = self.train(train_data_loader)
            valid_loss, valid_ndcg_k, valid_recall_k = self.validate(valid_data_loader)
            print(f"epoch: {epoch+1} train_loss: {train_loss}, valid_loss: {valid_loss}, valid_ndcg: {valid_ndcg_k}, valid_recall_k: {valid_recall_k}")
    
    
    def train(self, train_data_loader):
        self.model.train()
        total_loss = 0
        for i, data in tqdm(enumerate(train_data_loader)):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            pred = self.model(X)
            batch_loss = self.loss(pred, y)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss
        
        total_loss /= len(train_data_loader)
        return total_loss
            
        
    def validate(self, valid_data_loader):
        self.model.eval()
        valid_loss = 0
        total_X = []
        for i, data in tqdm(enumerate(valid_data_loader)):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            
            pred = self.model(X)
            batch_loss = self.loss(pred, y)
            valid_loss += batch_loss

            total_X.append(data['X'])

        total_X = np.concatenate(total_X, axis=0)

        valid_recall_k, valid_ndcg_k = self.evaluate(total_X)

        valid_loss /= len(valid_data_loader)

        return valid_loss, valid_recall_k, valid_ndcg_k
    
    
    # calcualte recall and ndcg
    def evaluate(self, X, k=10):
        self.model.eval()
        eval_recall_k = 0
        eval_ndcg_k = 0

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']

        actual = pd.DataFrame(X, columns = ['user', 'item']).groupby('user')['item'].agg(set).sort_index().to_dict() # user별 사용한 item list

        prediction = []
        for user in tqdm(range(num_users)):
            user_X = torch.tensor([[user, item] for item in range(num_items)], dtype=int).to(self.device)
            user_pred = self.model(user_X).detach().cpu()
            user_mask = torch.tensor([0 if item in actual[user] else 1 for item in range(num_items)], dtype=int)
            
            user_pred = user_pred.squeeze(1) * user_mask
        
            user_pred = np.argsort(user_pred.numpy())[::-1]
            prediction.append(user_pred)

        eval_recall_k = recall_at_k(list(actual.values()), prediction, k)
        eval_ndcg_k = ndcg_k(list(actual.values()), prediction, k)

        return eval_recall_k, eval_ndcg_k
