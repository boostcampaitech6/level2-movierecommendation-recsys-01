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
        self.train_actual = None
        self.valid_actual = None


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
            print(f"epoch: {epoch+1} train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid_ndcg: {valid_ndcg_k:.4f}, valid_recall_k: {valid_recall_k:.4f}")
    
    
    def train(self, train_data_loader):
        print("Training Start....")
        self.model.train()
        total_loss = 0
        total_X = []

        for i, data in enumerate(tqdm(train_data_loader)):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            pred = self.model(X)
            batch_loss = self.loss(pred, y)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss
            total_X.append(data['X'])
        
        total_loss /= len(train_data_loader)

        total_X = np.concatenate(total_X, axis=0)

        if self.train_actual is None:
            self.train_actual = self.actual_interaction_dict(total_X)

        return total_loss
            
        
    def validate(self, valid_data_loader):
        print("Validating Start....")
        self.model.eval()
        valid_loss = 0
        total_X = []

        for _, data in enumerate(tqdm(valid_data_loader)):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            
            pred = self.model(X)
            batch_loss = self.loss(pred, y)
            valid_loss += batch_loss

            total_X.append(data['X'])

        valid_loss /= len(valid_data_loader)

        total_X = np.concatenate(total_X, axis=0)
        if self.valid_actual is None:
            self.valid_actual = self.actual_interaction_dict(total_X) # valid 평가시엔 valid actual로
        valid_recall_k, valid_ndcg_k = self.evaluate()

        return valid_loss, valid_recall_k, valid_ndcg_k
    
    
    # calculate recall and ndcg
    def evaluate(self, k=10):
        print("Evaluating Start....")
        self.model.eval()
        eval_recall_k = 0
        eval_ndcg_k = 0

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        
        prediction = []
        print("Predict all users and items interaction....")
        for _, user in enumerate(tqdm(range(num_users))):
            user_X = torch.tensor([[user, item] for item in range(num_items)], dtype=int).to(self.device)
            user_mask = torch.tensor([0 if item in self.train_actual[user] else 1 for item in range(num_items)], dtype=int)
            
            user_pred = self.model(user_X).detach().cpu()
            user_pred = user_pred.squeeze(1) * user_mask # train interaction 제외
            
            user_pred = np.argpartition(user_pred.numpy(), -k)[-k:]
            prediction.append(user_pred)

        assert len(prediction) == num_users, f"prediction's length should be same as num_users({num_users}): {len(prediction)}"

        eval_recall_k = recall_at_k(list(self.valid_actual.values()), prediction, k)
        eval_ndcg_k = ndcg_k(list(self.valid_actual.values()), prediction, k)

        return eval_recall_k, eval_ndcg_k


    def actual_interaction_dict(self, X):
        return pd.DataFrame(X, columns = ['user', 'item']).groupby('user')['item'].agg(set).sort_index().to_dict()
