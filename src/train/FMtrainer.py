'''
trainer.py

- run (train and valid)
- evaluate (metric)
- train (only loss)
- valid (metric)
'''
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
import wandb

from ..models.DeepFMModels import DeepFM
from ..models.FMModels import FM
from ..metrics import recall_at_k, ndcg_k
from ..data.features import make_year

import logging
logger = logging.getLogger(__name__)


class FMTrainer():
    def __init__(self, args, evaluate_data, data_pipeline, runname) -> None:

        self.args = args
        self.device = torch.device(self.args.device)
        self.data_pipeline = data_pipeline
        self.num_features = self.data_pipeline.num_features
        self.cat_features_size = self.data_pipeline.cat_features_size
        self.runname = runname

        self.best_model_dir = f"{self.args.model_dir}/{runname}"
        Path(self.best_model_dir).mkdir(exist_ok=True, parents=True)

        if self.args.model_name == "FM":
            self.model = FM(self.num_features, self.cat_features_size, self.args.emb_dim)
        elif self.args.model_name == "DeepFM":
            self.model = DeepFM(self.num_features, self.cat_features_size, self.args.emb_dim, mlp_dims=[200, 200, 200], drop_rate=0.1)
        else:
            raise Exception

        self.model.to(self.device)
        self.loss = torch.nn.BCELoss()

        self.train_actual = None
        self.valid_actual = None
        self.total_interaction = torch.tensor(evaluate_data.values).to(self.device)

    def get_model(self):
        return self.model
    
       
    def run(self, train_data_loader, valid_data_loader):
        logger.info("Run Trainer...")
        patience = 10
        best_loss, best_epoch, endurance, best_ndcg_k, best_recall_k = 1e+9, 0, 0, 0, 0

        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.lr)
        else:
            logger.error(f"Optimizer Not Exists: {self.args.optimizer}")
            raise Exception

        def save_model_info(best_loss, best_epoch, train_loss, valid_recall_k, valid_ndcg_k):
            
            info = f'''\
            epoch:              {best_epoch}
            train_loss:         {train_loss}
            valid_loss:         {best_loss}
            valid Recall@10:    {valid_recall_k}
            valid nDCG@10:      {valid_ndcg_k}
            '''

            with open(f'{self.best_model_dir}/best_model_info.txt', 'w') as f:
                f.write(info)
            
        for epoch in range(self.args.epochs):
            train_loss = self.train(train_data_loader)
            valid_loss, valid_ndcg_k, valid_recall_k = self.validate(valid_data_loader)
            logger.info(f"epoch: {epoch+1} train_loss: {train_loss:.10f}, valid_loss: {valid_loss:.10f}, valid_ndcg: {valid_ndcg_k:.10f}, valid_recall_k: {valid_recall_k:.10f}")

            # wandb logging
            if self.args.wandb:
                wandb.log(
                    {
                        'epoch': (epoch+1),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'valid Recall@10': valid_recall_k,
                        'valid nDCG@10': valid_ndcg_k,
                    }
                )
            
            if valid_loss < best_loss:
                best_loss, best_epoch, best_ndcg_k, best_recall_k = valid_loss, epoch, valid_ndcg_k, valid_recall_k
                endurance = 1

                torch.save(self.model.state_dict(), f'{self.best_model_dir}/best_model.pt')
                save_model_info(best_loss, best_epoch, train_loss, best_recall_k, best_ndcg_k)
            else:
                endurance += 1
                if endurance >= patience:
                    break

        # wandb logging
        if self.args.wandb:
            wandb.log(
                {
                    'best_epoch': (best_epoch+1),
                    'best_loss': best_loss,
                    'best Recall@10': best_recall_k,
                    'best nDCG@10': best_ndcg_k,
                }
            )

    
    def train(self, train_data_loader):
        logger.info("Training Start....")
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

            total_loss += batch_loss.item()

            positive_index = torch.where(data['y'][:,0]==1)
            total_X.append(data['X'][positive_index])
        
        total_loss /= len(train_data_loader)
        total_X = np.concatenate(total_X, axis=0)

        if self.train_actual is None:
            self.train_actual = self.actual_interaction_dict(total_X)

        return total_loss
            
    def validate(self, valid_data_loader):
        logger.info("Start Validating....")
        self.model.eval()
        valid_loss = 0
        total_X = []

        for _, data in enumerate(tqdm(valid_data_loader)):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            pred = self.model(X)
            batch_loss = self.loss(pred, y)
            valid_loss += batch_loss.item()
            positive_index = torch.where(data['y'][:,0]==1)
            total_X.append(data['X'][positive_index])

        valid_loss /= len(valid_data_loader)

        total_X = np.concatenate(total_X, axis=0)

        if self.valid_actual is None:
            self.valid_actual = self.actual_interaction_dict(total_X) # valid 평가시엔 valid actual로

        valid_recall_k, valid_ndcg_k = self.evaluate()

        return valid_loss, valid_recall_k, valid_ndcg_k
    

    # calculate recall and ndcg
    def evaluate(self, k=10):
        logger.info("Start Evaluating....")
        self.model.eval()
        eval_recall_k = 0
        eval_ndcg_k = 0

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        offset = len(self.num_features)

        prediction = []
        
        logger.info("[EVAL]Predict all users and items interaction....")
        users = self.total_interaction[:, offset].unique().detach().cpu().numpy()
        for idx, user in enumerate(tqdm(users)):

            start_idx, end_idx = idx * num_items, (idx+1) * num_items
            user_X = self.total_interaction[start_idx:end_idx, :]
            user_items = user_X.detach().cpu().numpy()[:, offset+1]
            user_mask = torch.tensor([0 if item.item() in self.train_actual[int(user)] else 1 for item in user_items], dtype=int)

            user_pred = self.model(user_X.float()).detach().cpu()
            user_pred = user_pred.squeeze(1) * user_mask # train interaction 제외
            
            # find high prob index
            high_index = np.argpartition(user_pred.numpy(), -k)[-k:]
            # find high prob item by index
            user_recom = user_items[high_index]
            prediction.append(user_recom)

        assert len(prediction) == num_users, f"prediction's length should be same as num_users({num_users}): {len(prediction)}"

        eval_recall_k = recall_at_k(list(self.valid_actual.values()), prediction, k)
        eval_ndcg_k = ndcg_k(list(self.valid_actual.values()), prediction, k)

        return eval_recall_k, eval_ndcg_k


    def actual_interaction_dict(self, X):
        offset = len(self.num_features)
        return pd.DataFrame(X[:, offset:offset+2], columns = ['user', 'item']).groupby('user')['item'].agg(set).sort_index().to_dict()

    def inference(self, k=10):
        logger.info("Inference Start....")
        self.model.eval()

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        offset = len(self.num_features)

        prediction = []

        logger.info("[INFER]Predict all users and items interaction....")
        users = self.total_interaction[:, offset].unique().detach().cpu().numpy()
        for idx, user in enumerate(tqdm(users)):

            start_idx, end_idx = idx * num_items, (idx+1) * num_items
            user_X = self.total_interaction[start_idx:end_idx, :]
            user_items = user_X.detach().cpu().numpy()[:, offset+1]
            user_mask = torch.tensor([0 if (
                item.item() in self.train_actual[int(user)]) or (
                item.item() in self.valid_actual[int(user)]) else 1 for item in user_items], dtype=int)

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

    def load_best_model(self):
        logger.info('load best model...')
        self.model.load_state_dict(torch.load(f'{self.best_model_dir}/best_model.pt'))

        logger.info('best model info...')
        with open(f'{self.best_model_dir}/best_model_info.txt', 'r') as f:
            info = f.readlines()
        logger.info(info)
