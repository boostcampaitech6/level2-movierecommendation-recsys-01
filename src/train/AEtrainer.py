from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
import wandb

from ..models.AEModels import AE, VAE, MultiVAE
from ..metrics import recall_at_k, ndcg_k
from ..data.features import make_year
#from ..loss import VAELoss, loss_function_vae

import logging
logger = logging.getLogger(__name__)

class AETrainer():
    def __init__(self, args, evaluate_data, data_pipeline, runname) -> None:

        self.args = args
        self.device = torch.device(self.args.device)
        self.data_pipeline = data_pipeline
        self.num_features = self.data_pipeline.num_features
        self.cat_features_size = self.data_pipeline.cat_features_size
        self.runname = runname

        self.best_model_dir = f"{self.args.model_dir}/{runname}"
        Path(self.best_model_dir).mkdir(exist_ok=True, parents=True)

        if self.args.model_name == "AE":
            self.model = AE(self.data_pipeline.items.shape[0], self.args.latent_dim, self.args.encoder_dims)
#        elif self.args.model_name == "VAE":
#            self.model = VAE(evaluate_data['X'].shape[1], self.args.latent_dim, self.args.hidden_layers)
            #self.model = VAE(evaluate_data['X'].shape[1], self.args.latent_dim, self.args.hidden_layers)
            #self.model = MultiVAE(evaluate_data['X'].shape[1], self.args.p_dims)
        else:
            raise Exception

        self.model.to(self.device)
        if self.args.model_name == 'AE':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.args.model_name in ("VAE"):
            self.loss = torch.nn.BCEWithLogitsLoss()#loss_function_vae#VAELoss()
        else:
            raise Exception

        self.evaluate_data = evaluate_data

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
            train_loss, train_pred = self.train(train_data_loader)
            valid_loss = self.validate(train_pred, valid_data_loader)
            valid_recall_k, valid_ndcg_k = self.evaluate(train_data_loader, valid_data_loader)
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
        train_loss, train_pred = 0, []

        #for i, (train_data, valid_data) in enumerate(tqdm(train_data_loader, valid_data_loader)):
        for i, train_data in enumerate(tqdm(train_data_loader)):

            interact = train_data['interact'].to(self.device)
            all_mask = train_data['interact_all_mask'].to(self.device)

            pred = self.model(interact)

            # masking
            masked_pred = pred[all_mask] #pred * all_mask
            masked_interact = interact[all_mask] #interact * all_mask

            # loss
            batch_loss = self.loss(masked_pred, masked_interact)
            
            # backprop
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # sum of batch loss
            train_loss += batch_loss.item()
            train_pred.append(pred.detach().cpu())
        
        return train_loss/len(train_data_loader), train_pred

    def validate(self, train_pred, valid_data_loader):
        valid_loss = 0
        return valid_loss

    def evaluate(self, train_data_loader, valid_data_loader, k=10):
        recall_at_k = 0
        ndcg_at_k = 0
        return recall_at_k, ndcg_at_k

    def load_best_model(self):
        logger.info('load best model...')
        self.model.load_state_dict(torch.load(f'{self.best_model_dir}/best_model.pt'))

        logger.info('best model info...')
        with open(f'{self.best_model_dir}/best_model_info.txt', 'r') as f:
            info = f.readlines()
        logger.info(info)

    def inference(self, k=10):
        prediction = [0]
        return prediction
