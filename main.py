import torch
import pandas as pd
from options import args
from models import model_factory
from dataloaders import dataloader_factory, loader_submission
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    sub, inv_umap, inv_smap, user_lst = loader_submission(args)
    print(type(sub))
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

    inference_model = (input('Inference model? y/[n]: ')=='y')
    if inference_model:
        preds = trainer.submission(sub, inv_umap, inv_smap, user_lst)
        #print(preds)
        # Flatten the dictionary values
        # Create an empty list to store the flattened data
        flattened_data = []

        # Iterate over each key-value pair
        for key, values in preds.items():
        # Iterate over each value in the nested array
            for value in values:
                flattened_data.append((key, value))
        
        # Construct DataFrame
        df = pd.DataFrame(data=flattened_data)
        #df.drop(index=df[df['0']==6881][10:].index.tolist(), inplace=True)
        df.to_csv('out.csv', index=False)

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
