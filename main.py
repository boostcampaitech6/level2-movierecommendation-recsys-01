import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory, loader_submission
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    inv_umap, inv_smap = loader_submission(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

    inference_model = (input('Inference model? y/[n]: ')=='y')
    if inference_model:
        preds = trainer.submission(inv_umap, inv_smap)
        #generate_submission_file(args.data_file, preds)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
