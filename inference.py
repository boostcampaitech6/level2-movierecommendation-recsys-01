import argparse
import os

import torch
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
 
def submission(model):
        print("Make sumbission!")
        prd_list = None
        answer_list = None
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(submission_dataloader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                    
                seqs, candidates, labels = batch
                scores = model(seqs)  # B x T x V
                scores = scores[:, -1, :]  # B x V
                scores = scores.gather(1, candidates)  # B x C
                rank = (-scores).argsort(dim=1)
                cut = rank[:, :10]
        
                
        return preds

def main():
  # config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
  #  model_file='/opt/ml/input/code/BERT4Rec-VAE-Pytorch/experiments/test_2024-02-02_18/models/best_acc_model.pth',
#)  # Here you can replace it by your model path.

    model = models.bert(args=args)
    best_model = torch.load('/opt/ml/input/code/BERT4Rec-VAE-Pytorch/experiments/test_2024-02-02_18/models/best_acc_model.pth').get('model_state_dict')'#os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    #trainer.load(args.checkpoint_path)
    #print(f"Load model from {args.checkpoint_path} for submission!")
    preds = trainer.submission(model)
    generate_submission_file(args.data_file, preds)


if __name__ == "__main__":
    main()
