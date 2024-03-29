from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from ..models.AEModels import AE, DAE, VAE
from ..metrics import recall_at_k, ndcg_k
from ..loss import MultiAELoss, VAELoss

class AEInferencer:

    def __init__(self, args, evaluate_data, data_pipeline, runname):

        self.args = args
        self.device = torch.device(self.args.device)
        self.data_pipeline = data_pipeline
        self.runname = runname

        self.best_model_dir = f"{self.args.model_dir}/{runname}"
        Path(self.best_model_dir).mkdir(exist_ok=True, parents=True)

        self.evaluate_data = evaluate_data
        self._load_model()

    def _load_model(self):
        print('load best model...')
        if self.args.model_name in ("AE", "MultiAE"):
            self.model = AE(self.data_pipeline.items.shape[0], self.args.latent_dim, self.args.encoder_dims,
                self.args.dropout)
        elif self.args.model_name in ("DAE", "MultiDAE"):
            self.model = DAE(self.data_pipeline.items.shape[0], self.args.latent_dim, self.args.encoder_dims, 
                self.args.noise_factor, self.args.dropout)
        elif self.args.model_name in ("VAE", "MultiVAE"):
            self.model = VAE(self.data_pipeline.items.shape[0], self.args.latent_dim, self.args.encoder_dims,
                self.args.dropout)
        elif self.args.model_name in ("VDAE", "MultiVDAE"):
            self.model = VAE(self.data_pipeline.items.shape[0], self.args.latent_dim, self.args.encoder_dims,
                self.args.noise_factor, self.args.dropout, True)
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

    def inference(self, evaluate_data, k=10, extra_k=30):
        print("Inference Start....")
        self.model.eval()
        
        users = self.data_pipeline.users
        items = self.data_pipeline.items

        interact_tensor = torch.tensor(evaluate_data['interact']).to(self.device) 
        pos_mask_tensor = torch.tensor(evaluate_data['interact_pos_mask'])

        recommendation = []
        extra_recommendation, extra_proba = [],[]

        for i, user in enumerate(tqdm(users)): 
            user_interact = interact_tensor[i,:]
            inv_pos_mask = ~(pos_mask_tensor[i,:])

            # prediction
            if self.args.model_name in ('VAE', 'MultiVAE', 'VDAE', 'MultiVDAE'):
                user_pred, _, _ = self.model(user_interact)
                user_pred = user_pred.detach().cpu()
            else:
                user_pred = self.model(user_interact).detach().cpu()

            user_pred = (user_pred * inv_pos_mask).numpy()

            # find high prob index
            high_index = np.argpartition(user_pred, -k)[-k:]

            # find high prob item by index
            user_recom = items[high_index]

            # save
            recommendation.append(user_recom)

            # find extra high prob and high prob index
            if extra_k is not None:
                extra_high_index = np.argpartition(user_pred, -extra_k)[-extra_k:]
                extra_high_prob = user_pred[extra_high_index]

                extra_user_recom = items[extra_high_index]

                extra_recommendation.append(extra_user_recom)
                extra_proba.append(extra_high_prob)

        # expand_dims
        recommendation = np.expand_dims(np.concatenate(recommendation, axis=0), axis=-1)
        user_ids = np.expand_dims(np.repeat(users, 10), axis=-1).astype(int)

        recommendation = np.concatenate([user_ids, recommendation], axis=1)

        if extra_k is not None:
            extra_recommendation = np.expand_dims(np.concatenate(extra_recommendation, axis=0), axis=-1)
            extra_proba = np.expand_dims(np.concatenate(extra_proba, axis=0), axis=-1)
            user_ids = np.expand_dims(np.repeat(users, extra_k), axis=-1).astype(int)

            extra_recommendation = np.concatenate([user_ids, extra_recommendation, extra_proba], axis=1)
        else:
            extra_recommendation = None

        return recommendation, extra_recommendation
