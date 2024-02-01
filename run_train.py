'''
run_train.py

- args
- set seed
- wandb setting
- load data
- dataloader
- train
- evaluation
'''
import os
from src.data.datasets import (
    get_data, split_data, save_data, load_data, FMDataset)

def main():
    # args
    # data version으로 관리한다면?
    
    # get data - version
    train_name = 'train_data.pickle'
    valid_name = 'valid_data.pickle'
    
    if not os.path.exists(train_name):
        data = get_data()
        train_data, valid_data = split_data(data)

        save_data(train_data, train_name)
        save_data(valid_data, valid_name)
    else:
        train_data = load_data(train_name)
        valid_data = load_data(valid_name)
    
    # dataset
    train_dataset = FMDataset(train_data, train=True)
    valid_dataset = FMDataset(valid_data, train=True)
#
#    # dataloader
#    train_dataloader = DataLoader(train_dataset, shuffle=True)
#    valid_dataloader = DataLoader(valid_dataset, shuffle=True)


if __name__ == '__main__':
    main()
