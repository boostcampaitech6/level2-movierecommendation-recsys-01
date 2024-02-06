from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML2MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-2m'

    @classmethod
    def url(cls):
        return 'https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000176/data/data.tar.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['train_ratings.csv',
                'directors.tsv',
                'genres.tsv',
                'titles.tsv',
                'writers.tsv',
                'years.tsv',
                ]
    @classmethod
    def is_zipfile(cls):
        return False

    def load_ratings_df(self):
        #breakpoint()
        folder_path = self._get_rawdata_folder_path()
        print(folder_path)
        file_path = folder_path.joinpath('data','train','train_ratings.csv')
        print(file_path)
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid','timestamp']
        return df


