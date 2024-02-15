import re
import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)


def make_year(df, data_path='/data/ephemeral/data/train', debug=False):
    logger.info("[FE] make year...")
    # read file
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    
    # mapping new feature
    year_map = year_data.groupby('item').agg('max').to_dict()['year']
    df['year'] = df['item'].map(year_map)
    
    # handle Nan and anomal value
    df = _fill_year_with_title_info(df, debug)

    assert df['year'].isna().sum() == 0, f"year has Nan value"

def _fill_year_with_title_info(df, debug=False):
    make_title(df)
    logger.info("fill nan year with title info...")

    if debug:
        print(df.head())

    def year_at_title(title):
        start, end = title.rfind('(') + 1, title.rfind(')')
        return title[start:end].split('-')[0]

    df['year'].fillna(df['title'].apply(year_at_title).astype(float), inplace=True)
    df['year'] = df['year'].astype(int)

    return df

def make_writer(df, data_path='/data/ephemeral/data/train'):
    logger.info("[FE] make writer...")
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    writer_map = writer_data.groupby('item').agg('max').to_dict()['writer']
    df['writer'] = df['item'].map(writer_map)


def make_title(df, data_path='/data/ephemeral/data/train'):
    logger.info("[FE] make title...")
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    title_map = title_data.groupby('item').agg('max').to_dict()['title']
    df['title'] = df['item'].map(title_map)


def make_genre(df, data_path='/data/ephemeral/data/train'):
    logger.info("[FE] make genre...")
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    genre_map = genre_data.groupby('item')['genre'].apply(list).to_dict()
    df['genre'] = df['item'].map(genre_map)


def make_director(df, data_path='/data/ephemeral/data/train'):
    logger.info("[FE] make director...")
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
    director_map = director_data.groupby('item').agg('max').to_dict()['director']
    df['director'] = df['item'].map(director_map)

# def add_year(tensor, data_pipeline, data_path='/data/ephemeral/data/train'):
#     item = tensor[:, 1].detach().cpu().numpy()

#     title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
#     title_map = title_data.groupby('item').agg('max').to_dict()['title']

#     def extract_year(title):
#         # 정규 표현식을 사용하여 연도 정보 추출
#         year_match = re.search(r'\((\d{4})(?:\s*-\s*\d{4})?\)', title)
#         if year_match:
#             return int(year_match.group(1))
#         else:
#             raise ValueError
        
#     year = torch.tensor([extract_year(title_map[data_pipeline.decode_categorical_features(item_id)]) for item_id in item])
    
#     return torch.cat((tensor, year), axis=1)