'''
trainer.py

- run (train and valid)
- evaluate (metric)
- train (only loss)
- valid (metric)
'''
from tqdm import tqdm
import torch
from .models.FMModels import FM
from .models.DeepFMModels import DeepFM

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
            print(f"epoch: {epoch+1} train_loss: {train_loss}, valid_loss: {valid_loss}, 
                  valid_ndcg: {valid_ndcg_k}, valid_recall_k: {valid_recall_k}")
    
    
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
        return total_loss
            
        
    def validate(self, valid_data_loader):
        self.model.eval()
        return None, None, None
    
    
    def evaluate(self):
        self.model.eval()
        pass
    
    
    
    
    