# datasets.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

def neg_sampling(train_df, num_users, num_items, num_negs):
    
    # positive 파악
    user_items = train_df.groupby('user').agg(pos_items=('item', set))

    # positive 빼기
    neg_items = user_items['pos_items'].apply(lambda x: set(range(num_items))-x)

    # sampling
    print('negative sampling...')
    user_items['neg_items'] = neg_items.apply(
            lambda x: np.random.choice(list(x), size=num_negs))
    print('done')
    
    # series to dataframe
    # user, item
    neg_samples_df = user_items.explode('neg_items')
    neg_samples_df = neg_samples_df.drop(columns=['pos_items']).reset_index()
    neg_samples_df = neg_samples_df.rename(columns={'neg_items': 'item'})
    neg_samples_df['rating'] = 0

    return neg_samples_df

def split_data(data):
    # entry-based split
    train_data, valid_data = None, None
    return train_data, valid_data

def get_data():

    # read csv
    train_path = '/data/ephemeral/data/train/train_ratings.csv'
    train_df = pd.read_csv(train_path)

    # drop time
    train_df = train_df.drop(columns=['time'])

    # num users and items
    num_users = train_df.user.nunique()
    num_items = train_df.item.nunique()

    # ordinal encoding
    cat_features = ['user', 'item']
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder = ordinal_encoder.fit(train_df[cat_features])
    train_df[cat_features] = ordinal_encoder.transform(train_df[cat_features])
    cat_features_size = {cat_name: len(categories) for cat_name, categories in zip(
        cat_features, ordinal_encoder.categories_)}

    # astype
    train_df[cat_features] = train_df[cat_features].astype(int)

    # rating
    train_df['rating'] = 1

    # neg sample
    num_negs = 5
    neg_samples_df = neg_sampling(train_df, num_users, num_items, num_negs)
    train_df = pd.concat([train_df, neg_samples_df], axis=0)

    # shuffle data
    train_df.shuffle()

    data = {
        'train_df': train_df,
        'cat_features_size': cat_features_size,
    }

    return data
