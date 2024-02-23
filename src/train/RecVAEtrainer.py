from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
import wandb

#from ..models.AEModels import AE, DAE, VAE
from ..models.RecVAEModels import RecVAE
from ..metrics import recall_at_k, ndcg_k
#from ..loss import MultiAELoss, VAELoss, ConfidenceAELoss, MultiConfidenceAELoss

import logging
logger = logging.getLogger(__name__)

class RecVAETrainer():
    def __init__(self, args, evaluate_data, data_pipeline, runname) -> None:

        self.args = args
        self.device = torch.device(self.args.device)
        self.data_pipeline = data_pipeline
        self.num_features = self.data_pipeline.num_features
        self.cat_features_size = self.data_pipeline.cat_features_size
        self.runname = runname

        self.best_model_dir = f"{self.args.model_dir}/{runname}"
        Path(self.best_model_dir).mkdir(exist_ok=True, parents=True)

        if self.args.model_name == 'RecVAE':
            if self.args.beta == 'None': self.args.beta = None
            if self.args.gamma == 'None': self.args.gamma = None
            self.model = RecVAE(self.data_pipeline.items.shape[0], self.args.hidden_dim, self.args.latent_dim,
                self.args.dropout, self.args.beta, self.args.gamma)
        else:
            raise Exception

        self.confidence_standard = self.args.confidence if self.args.confidence != 'None' else None
        self.model.to(self.device)

        # loss (?)
#        if (self.args.model_name in ('AE', 'DAE')) and self.confidence_standard:
#            print('Confidence Loss used')
#            self.loss = ConfidenceAELoss()
#            self.confidence_array = torch.tensor(self.data_pipeline.confidence_array).to(self.device)
#        elif self.args.model_name in ('AE', 'DAE'):
#            self.loss = torch.nn.BCEWithLogitsLoss()
#        elif self.args.model_name in ('MultiAE', 'MultiDAE') and self.confidence_standard:
#            print('Confidence Loss used')
#            self.loss = MultiConfidenceAELoss()
#            self.confidence_array = torch.tensor(self.data_pipeline.confidence_array).to(self.device)
#        elif self.args.model_name in ('MultiAE', 'MultiDAE'):
#            self.loss = MultiAELoss()
#        elif self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#            self.loss = VAELoss(args)
#        else:
#            raise Exception
#
        self.evaluate_data = evaluate_data

    def get_model(self):
        return self.model

    def run(self, train_data_loader, valid_data_loader, valid_data):
        logger.info("Run Trainer...")
        patience = 10 
        best_loss, best_epoch, endurance, best_ndcg_k, best_recall_k = 1e+9, 0, 0, 0, 0

        decoder_params = set(self.model.decoder.parameters())
        encoder_params = set(self.model.encoder.parameters())

        if self.args.optimizer == "adamw":
            self.encoder_optimizer = torch.optim.AdamW(encoder_params, lr = self.args.lr)
            self.decoder_optimizer = torch.optim.AdamW(decoder_params, lr = self.args.lr)
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

            if self.args.not_alternating:
                train_loss = self.train(train_data_loader, [self.encoder_optimizer, self.decoder_optimizer], 1)
            else:
                _ = self.train(train_data_loader, [self.decoder_optimizer], self.args.encoder_n_epochs)
                self.model.update_prior()
                train_loss = self.train(train_data_loader, [self.decoder_optimizer], self.args.decoder_n_epochs)

            valid_loss = self.validate(valid_data_loader)
            valid_recall_k, valid_ndcg_k = self.evaluate(train_data_loader, valid_data)
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
            # if valid_recall_k > best_recall_k:
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

    def train(self, train_data_loader, opts, n_epochs):
    # def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
        logger.info("Training Start....")
        self.model.train()
        train_loss = 0
        #model.train()

        for epoch in range(n_epochs):
            for i, train_data in enumerate(tqdm(train_data_loader)):

                interact = train_data['interact'].to(self.device)
                all_mask = train_data['interact_all_mask'].to(self.device)

#        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
#            ratings = batch.get_ratings_to_dev()

                for optimizer in opts:
                    optimizer.zero_grad()
                    
                _, loss = self.model(interact, all_mask)
                loss.backward()
                
                for optimizer in opts:
                    optimizer.step()

                # sum of batch loss
                train_loss += loss.item()

        return train_loss/len(train_data_loader)/n_epochs


#    def _train(self, train_data_loader):
#        logger.info("Training Start....")
#        self.model.train()
#        train_loss, train_pred = 0, []
#
#        for i, train_data in enumerate(tqdm(train_data_loader)):
#
#            interact = train_data['interact'].to(self.device)
#            all_mask = train_data['interact_all_mask'].to(self.device)
#            
#            if self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#                pred, mean, logvar = self.model(interact)
#            else:
#                pred = self.model(interact)
#
#            # masking
#            masked_pred = pred[all_mask] #pred * all_mask
#            masked_interact = interact[all_mask] #interact * all_mask
#
#            # loss
#            if self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#                batch_loss = self.loss(masked_pred, masked_interact, mean, logvar, True)
#            elif self.args.model_name in ('AE', 'DAE', 'MultiAE', 'MultiDAE') and self.confidence_standard:
#                confidence = self.confidence_array[i*interact.size(0):(i+1)*interact.size(0),:]
#                masked_confidence = confidence[all_mask]
#                batch_loss = self.loss(masked_pred, masked_interact, masked_confidence)
#            else:
#                batch_loss = self.loss(masked_pred, masked_interact)
#            
#            # backprop
#            self.optimizer.zero_grad()
#            batch_loss.backward()
#            self.optimizer.step()
#            
#            # sum of batch loss
#            train_loss += batch_loss.item()
#            if self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#                train_pred.append([pred.detach().cpu(), mean.detach().cpu(), logvar.detach().cpu()])
#            else:
#                train_pred.append(pred.detach().cpu())
#        
#        return train_loss/len(train_data_loader), train_pred

    def validate(self, valid_data_loader):
        logger.info("Start Validating....")
        self.model.eval()
        valid_loss = 0

        for i, valid_data in enumerate(tqdm(valid_data_loader)):

            interact = valid_data['interact'].to(self.device)
            all_mask = valid_data['interact_all_mask'].to(self.device)

            _, loss = self.model(interact, all_mask)

            # sum of batch loss
            valid_loss += loss.item()

        return valid_loss/len(valid_data_loader)

#    def _validate(self, train_pred, valid_data_loader):
#        logger.info("Start Validating....")
#        self.model.eval()
#        valid_loss = 0
#
#        for i, (train_pred, valid_data) in enumerate(tqdm(zip(train_pred, valid_data_loader), total=len(train_pred))):
#            
#            if self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#                train_pred, train_mean, train_logvar = train_pred
#            train_pred = train_pred.to(self.device)
#
#            valid_interact = valid_data['interact'].to(self.device)
#            valid_all_mask = valid_data['interact_all_mask'].to(self.device)
#
#            # masking
#            masked_pred = train_pred[valid_all_mask] #pred * all_mask
#            masked_interact = valid_interact[valid_all_mask] #interact * all_mask
#
#            # loss
#            if self.args.model_name in ("VAE", 'MultiVAE', "VDAE", "MultiVDAE"):
#                batch_loss = self.loss(masked_pred, masked_interact, train_mean, train_logvar, False)
#            elif self.args.model_name in ('AE', 'DAE', 'MultiAE', 'MultiDAE') and self.confidence_standard:
#                confidence = self.confidence_array[i*valid_interact.size(0):(i+1)*valid_interact.size(0),:]
#                masked_confidence = confidence[valid_all_mask]
#                batch_loss = self.loss(masked_pred, masked_interact, masked_confidence)
#            else:
#                batch_loss = self.loss(masked_pred, masked_interact)
#            
#            # sum of batch loss
#            valid_loss += batch_loss.item()
#
#        return valid_loss/len(valid_data_loader)

    def evaluate(self, train_data_loader, valid_data, k=10):
        logger.info("Start Evaluating....")
        self.model.eval()
        valid_pos_items = valid_data['pos_items']
        user_items = self.data_pipeline.items # unique suer items
        prediction = []

        for i, train_data in enumerate(tqdm(train_data_loader)):

            interact = train_data['interact'].to(self.device)
            pos_mask = train_data['interact_pos_mask'].to(self.device)

            train_pred = self.model(interact, calculate_loss=False)

            # masking
            inv_train_pos_pred = (train_pred * ~pos_mask).detach().cpu()
            
            # find high prob index
            high_index = np.argpartition(inv_train_pos_pred.numpy(), -k)[:,-k:]

            # find high prob item by index
            user_recom = user_items[high_index]
            prediction.append(user_recom)

        prediction = np.concatenate(prediction, axis=0)

        eval_recall_at_k = recall_at_k(list(valid_pos_items.values()), prediction, k)
        eval_ndcg_at_k = ndcg_k(list(valid_pos_items.values()), prediction, k)

        return eval_recall_at_k, eval_ndcg_at_k

    def load_best_model(self):
        logger.info('load best model...')
        self.model.load_state_dict(torch.load(f'{self.best_model_dir}/best_model.pt'))

        logger.info('best model info...')
        with open(f'{self.best_model_dir}/best_model_info.txt', 'r') as f:
            info = f.readlines()
        logger.info(info)

    def inference(self, evaluate_data, k=10):
        logger.info("Inference Start....")
        self.model.eval()
        
        users = self.data_pipeline.users
        items = self.data_pipeline.items

        interact_tensor = torch.tensor(evaluate_data['interact']).to(self.device) 
        pos_mask_tensor = torch.tensor(evaluate_data['interact_pos_mask'])

        prediction = []

        for i, user in enumerate(tqdm(users)): 
            user_interact = interact_tensor[i,:].unsqueeze(0)
            inv_pos_mask = ~(pos_mask_tensor[i,:])

            # prediction
            user_pred = self.model(user_interact, calculate_loss=False).detach().cpu().squeeze()

            # masking
            user_pred = user_pred * inv_pos_mask
            
            # find high prob index
            high_index = np.argpartition(user_pred.numpy(), -k)[-k:]

            # find high prob item by index
            user_recom = items[high_index]
            prediction.append(user_recom)

        # expand_dims
        prediction = np.expand_dims(np.concatenate(prediction, axis=0), axis=-1)
        user_ids = np.expand_dims(np.repeat(users, 10), axis=-1).astype(int)

        prediction = np.concatenate([user_ids, prediction], axis=1)
        return prediction

