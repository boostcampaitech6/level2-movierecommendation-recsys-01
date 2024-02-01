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
from src.data.datasets import get_data

def main():
    # args
    # data version으로 관리한다면?
    
    # get data - version
    data = get_data()
    train_data, valid_data = split_data(data)
    
#    # dataset
#    train_dataset = FMDataset(train_data)
#    valid_dataset = FMDataset(valid_data)
#
#    # dataloader
#    train_dataloader = DataLoader(train_dataset, shuffle=True)
#    valid_dataloader = DataLoader(valid_dataset, shuffle=True)


if __name__ == '__main__':
    main()
