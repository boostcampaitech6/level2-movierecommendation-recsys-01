import re
import pandas as pd
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


def feature(serial_number, feature_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # assert before calling
            (origin, target) = args
            if origin is None:
                origin = target
            args = (origin, target)

            # validate existance
            if feature_name in target.columns:
                logger.info(f"\'#{serial_number}. {feature_name}\' is already in DataFrame...")
                return target
            # logging
            logger.info(f'[FE] make #{serial_number}. {feature_name}...')

            # main function call
            result = func(*args, **kwargs)

            # assert after calling
            assert (result is not None) and (isinstance(result, pd.DataFrame)), "You should return DataFrame"
            assert result[feature_name].isna().sum() == 0, f"{feature_name} has NaN value"
            return result
        return wrapper
    return decorator


def _ordered_set_tuple(series: pd.Series):
    '''
    writer, genre, director 와 같이 하나의 아이템에 여러개의 항목이 매핑되는 feature의 경우,
    각 조합을 하나의 entity로 취급하기 위함
    '''
    result = list(series.unique())
    result.sort()
    return str(tuple(result))


@feature(4, 'year')
def year(origin, target, data_path='/data/ephemeral/data/train'):
    # read file
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    
    # mapping new feature
    year_map = year_data.groupby('item').agg('max').to_dict()['year']
    target['year'] = target['item'].map(year_map)
    
    # handle Nan and anomal value
    def fill_year_with_title_info(origin, df):
        logger.info("You need feature \'title\' to fill NaN value in director...")
        title(origin, df)

        logger.info("fill nan year with title info...")
        def year_at_title(title):
            start, end = title.rfind('(') + 1, title.rfind(')')
            return title[start:end].split('-')[0]

        df['year'].fillna(df['title'].apply(year_at_title).astype(float), inplace=True)
        df['year'] = df['year'].astype(np.int32)

        return df
    target = fill_year_with_title_info(origin, target)
    return target


@feature(5, 'writer')
def writer(origin, df, data_path='/data/ephemeral/data/train'):
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    writer_map = writer_data.groupby('item').agg(_ordered_set_tuple).to_dict()['writer']
    df['writer'] = df['item'].map(writer_map)
    
    # fill nan with director if hybrid
    if 'director' not in df.columns:
        logger.info("You need feature \'director\' to fill NaN value in writer...")
        director(origin, df)
    
    writers = set(df['writer'].unique())
    directors = set(df['director'].unique())
    both = writers & directors
    if np.nan in both:
        both.remove(np.nan)

    def fill_with_both(id):
        if id in both:
            return id
        return 'nm9999999'
    nan_writer = df.writer.isna()
    df.loc[nan_writer, 'writer'] = df.loc[nan_writer, 'director'].apply(fill_with_both).astype('category')

    # 고유한 카테고리 값 추출
    categories = df['writer'].unique()

    # 각 카테고리를 숫자로 매핑하는 딕셔너리 생성
    category_to_num = {category: i for i, category in enumerate(categories)}

    # 매핑 딕셔너리를 사용하여 'genre' 열을 숫자로 변환
    df['writer'] = df['writer'].map(category_to_num).astype(np.int32)
    return df


@feature(6, 'title')
def title(origin, df, data_path='/data/ephemeral/data/train'):
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    title_map = title_data.groupby('item').agg('max').to_dict()['title']
    df['title'] = df['item'].map(title_map).astype('category')
    return df


@feature(7, 'genre')
def genre(origin, df, data_path='/data/ephemeral/data/train'):
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    genre_map = genre_data.groupby('item')['genre'].apply(_ordered_set_tuple).to_dict()
    df['genre'] = df['item'].map(genre_map).astype('category')

    # 고유한 카테고리 값 추출
    categories = df['genre'].unique()

    # 각 카테고리를 숫자로 매핑하는 딕셔너리 생성
    category_to_num = {category: i for i, category in enumerate(categories)}

    # 매핑 딕셔너리를 사용하여 'genre' 열을 숫자로 변환
    df['genre'] = df['genre'].map(category_to_num).astype(np.int32)
    return df


@feature(8, 'director')
def director(origin, df, data_path='/data/ephemeral/data/train'):
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
    director_map = director_data.groupby('item').agg(_ordered_set_tuple).to_dict()['director']
    df['director'] = df['item'].map(director_map)

    # fill nan with writer if hybrid
    if 'writer' not in df.columns:
        logger.info("You need feature \'writer\' to fill NaN value in director...")
        writer(origin, df)
    
    writers = set(df['writer'].unique())
    directors = set(df['director'].unique())
    both = writers & directors
    if np.nan in both:
        both.remove(np.nan)

    def fill_with_both(id):
        if id in both:
            return id
        return 'nm9999999'
    nan_director = df.director.isna()
    df.loc[nan_director, 'director'] = df.loc[nan_director, 'writer'].apply(fill_with_both).astype('category')

    # 고유한 카테고리 값 추출
    categories = df['director'].unique()

    # 각 카테고리를 숫자로 매핑하는 딕셔너리 생성
    category_to_num = {category: i for i, category in enumerate(categories)}

    # 매핑 딕셔너리를 사용하여 'genre' 열을 숫자로 변환
    df['director'] = df['director'].map(category_to_num).astype(np.int32)

    return df


@feature(9, 'era')
def era(origin, df):
    if 'year' not in df.columns:
        year(origin, df)
    
    def era_map(x: int) -> int:
        x = int(x)
        if x < 1920:
            return 1
        elif x >= 1920 and x < 1950:
            return 2
        elif x >= 1950 and x < 1970:
            return 3
        elif x >= 1970 and x < 1980:
            return 4
        elif x >= 1980 and x < 2000:
            return 5
        else:
            return 6

    df['era'] = df['year'].apply(era_map).astype(np.int32)
    return df


@feature(10, 'user_review_count')
def user_review_count(origin, df):
    user_review_count = origin.user.value_counts().to_dict()
    df['user_review_count'] = df['user'].map(user_review_count)
    df.user_review_count.fillna(0, inplace=True)
    df.user_review_count = np.log(df.user_review_count).astype(np.float32)
    return df


@feature(11, 'user_movie_count')
def user_movie_count(origin, df):
    user_movie_count = origin[['user', 'item']].groupby('user')['item'].agg('count').to_dict()
    df['user_movie_count'] = df['user'].map(user_movie_count)
    df.user_movie_count.fillna(0, inplace=True)
    df.user_movie_count = np.log(df.user_movie_count).astype(np.float32)
    return df


@feature(12, 'movie_user_count')
def movie_user_count(origin, df):
    movie_user_count = origin[['user', 'item']].groupby('item')['user'].agg('count').to_dict()
    df['movie_user_count'] = df['item'].map(movie_user_count)
    df.movie_user_count.fillna(0, inplace=True)
    df.movie_user_count = np.log(df.movie_user_count).astype(np.float32)
    return df


@feature(13, 'movie_user_count_category')
def movie_user_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['movie_user_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['movie_user_count_category'] = pd.cut(df['movie_user_count'], bins=bins, labels=labels)
    return df


@feature(14, 'user_movie_count_category')
def user_movie_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['user_movie_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['user_movie_count_category'] = pd.cut(df['user_movie_count'], bins=bins, labels=labels)
    return df


@feature(15, 'item_review_count')
def item_review_count(origin, df):
    item_review_count = origin.item.value_counts().to_dict()
    df['item_review_count'] = df['item'].map(item_review_count)
    df.item_review_count.fillna(0, inplace=True)
    df.item_review_count = np.log(df.item_review_count).astype(np.float32)
    return df


@feature(16, 'item_review_count_category')
def item_review_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['item_review_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['item_review_count_category'] = pd.cut(df['item_review_count'], bins=bins, labels=labels)
    return df


@feature(17, 'last_review_time')
def last_review_time(origin, df):
    last_review_time = origin[['user', 'time']].groupby('user')['time'].agg('max').to_dict()
    df['last_review_time'] = df['user'].map(last_review_time)
    df.last_review_time.fillna(0, inplace=True)
    return df


@feature(18, 'first_review_time')
def first_review_time(origin, df):
    first_review_time = origin[['user', 'time']].groupby('user')['time'].agg('min').to_dict()
    df['first_review_time'] = df['user'].map(first_review_time)
    df.first_review_time.fillna(0, inplace=True)
    return df


@feature(19, 'oldest_year')
def oldest_year(origin, df):
    oldest_year = origin[['user', 'year']].groupby('user')['year'].agg('min').to_dict()
    df['oldest_year'] = df['user'].map(oldest_year).astype(np.float32)
    df.oldest_year.fillna(0, inplace=True)
    return df


@feature(20, 'newest_year')
def newest_year(origin, df):
    newest_year = origin[['user', 'year']].groupby('user')['year'].agg('max').to_dict()
    df['newest_year'] = df['user'].map(newest_year).astype(np.float32)
    df.newest_year.fillna(0, inplace=True)
    return df


@feature(21, 'highest_year')
def highest_year(origin, df):
    highest_year = origin[['user', 'year']].groupby('user')['year'].agg(lambda x: x.mode()[0]).to_dict()
    df['highest_year'] = df['user'].map(highest_year).astype(np.float32)
    df.newest_year.fillna(0, inplace=True)
    return df


@feature(22, 'year_review_count')
def year_review_count(origin, df):
    year_review_count = origin.year.value_counts().to_dict()
    df['year_review_count'] = df['year'].map(year_review_count)
    df.year_review_count.fillna(0, inplace=True)
    df.year_review_count = np.log(df.year_review_count).astype(np.float32)
    return df


@feature(23, 'year_user_count')
def year_user_count(origin, df):
    year_user_count = origin[['year', 'user']].groupby('year')['user'].agg('count').to_dict()
    df['year_user_count'] = df['year'].map(year_user_count)
    df.year_user_count.fillna(0, inplace=True)
    df.year_user_count = np.log(df.year_user_count).astype(np.float32)
    return df


@feature(24, 'year_user_count_category')
def year_user_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['year_user_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['year_user_count_category'] = pd.cut(df['year_user_count'], bins=bins, labels=labels)
    return df


@feature(25, 'year_movie_count')
def year_movie_count(origin, df):
    year_movie_count = origin[['year', 'item']].groupby('year')['item'].agg('count').to_dict()
    df['year_movie_count'] = df['year'].map(year_movie_count)
    df.year_movie_count.fillna(0, inplace=True)
    df.year_movie_count = np.log(df.year_movie_count).astype(np.float32)
    return df


@feature(26, 'year_movie_count_category')
def year_movie_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['year_movie_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['year_movie_count_category'] = pd.cut(df['year_movie_count'], bins=bins, labels=labels)
    return df


@feature(27, 'time_diff')
def time_diff(origin: pd.DataFrame, df: pd.DataFrame):
    logger.info("fill NaN in time feature...")

    if 'time' not in df.columns:
        origin['time_key'] = 'u' + origin['user'].astype(str) +  'i' + origin['item'].astype(str)
        df['time_key'] = 'u' + df['user'].astype(str) + 'i' + df['item'].astype(str)

        time_key = origin[['time_key', 'time']].groupby('time_key')['time'].agg('min').to_dict()

        df['time'] = df['time_key'].map(time_key)
        del df.time_key, origin, time_key

    if 'year' not in df.columns:
        year(origin, df)
    
    min_time = df.time.min()
    df.time.fillna(min_time, inplace=True)

    # df['year'] 열을 datetime 타입으로 변환하여 'open_time' 열로 저장
    df['open_time'] = pd.to_datetime(df['year'], format='%Y')
    # 'open_time' 열의 값을 UTC 타임스탬프로 변환
    df['open_time'] = df['open_time'].astype(int) // 10**9

    # 'time' 열과 'open_time' 열의 차이를 계산하여 'time_diff' 열로 저장
    df['time_diff'] = df['time'] - df['open_time']
    df.loc[df.time_diff < 0, 'time_diff'] = 0

    df.time_diff = df.time_diff.astype(np.float32)
    return df


@feature(28, 'date')
def date(origin, df):
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df


@feature(29, 'review_year')
def review_year(origin, df):
    df['review_year'] = df['date'].dt.year
    return df


@feature(30, 'month')
def month(origin, df):
    df['month'] = df['date'].dt.month
    return df


@feature(31, 'day')
def day(origin, df):
    df['day'] = df['date'].dt.day
    return df


@feature(32, 'day_of_week')
def day_of_week(origin, df):
    df['day_of_week'] = df['date'].dt.day_name()
    return df


@feature(33, 'movie_first_review_year')
def movie_first_review_year(origin, df):
    movie_first_review_year = origin[['item', 'review_year']].groupby('item')['review_year'].agg('min').to_dict()
    df['movie_first_review_year'] = df['item'].map(movie_first_review_year)
    df.movie_first_review_year.fillna(0, inplace=True)
    df.movie_first_review_year = np.log(df.movie_first_review_year).astype(np.float32)
    return df


@feature(34, 'movie_last_review_year')
def movie_last_review_year(origin, df):
    movie_last_review_year = origin[['item', 'review_year']].groupby('item')['review_year'].agg('max').to_dict()
    df['movie_last_review_year'] = df['item'].map(movie_last_review_year)
    df.movie_last_review_year.fillna(0, inplace=True)
    df.movie_last_review_year = np.log(df.movie_last_review_year).astype(np.float32)
    return df


@feature(35, 'movie_most_review_year')
def movie_most_review_year(origin, df):
    movie_most_review_year = origin[['item', 'review_year']].groupby('item')['review_year'].agg(lambda x: x.mode()[0]).to_dict()
    df['movie_most_review_year'] = df['item'].map(movie_most_review_year)
    df.movie_most_review_year.fillna(0, inplace=True)
    df.movie_most_review_year = np.log(df.movie_most_review_year).astype(np.float32)
    return df


@feature(36, 'review_year_movie_count')
def review_year_movie_count(origin, df):
    review_year_movie_count = origin[['item', 'review_year']].groupby('review_year')['item'].agg('count').to_dict()
    df['review_year_movie_count'] = df['review_year'].map(review_year_movie_count)
    df.review_year_movie_count.fillna(0, inplace=True)
    df.review_year_movie_count = np.log(df.review_year_movie_count).astype(np.float32)
    return df


@feature(37, 'review_year_movie_count_category')
def review_year_movie_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['review_year_movie_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['review_year_movie_count_category'] = pd.cut(df['review_year_movie_count'], bins=bins, labels=labels)
    return df


@feature(38, 'user_writer_count')
def user_writer_count(origin, df):
    user_writer_count = origin[['user', 'writer']].groupby('user')['writer'].agg('count').to_dict()
    df['user_writer_count'] = df['user'].map(user_writer_count)
    df.user_writer_count.fillna(0, inplace=True)
    df.user_writer_count = np.log(df.user_writer_count).astype(np.float32)
    return df


@feature(39, 'writer_user_count')
def writer_user_count(origin, df):
    writer_user_count = origin[['user', 'writer']].groupby('writer')['user'].agg('count').to_dict()
    df['writer_user_count'] = df['writer'].map(writer_user_count)
    df.writer_user_count.fillna(0, inplace=True)
    df.writer_user_count = np.log(df.writer_user_count).astype(np.float32)
    return df


@feature(40, 'user_writer_count_category')
def user_writer_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['user_writer_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['user_writer_count_category'] = pd.cut(df['user_writer_count'], bins=bins, labels=labels)
    return df


@feature(41, 'writer_user_count_category')
def writer_user_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['writer_user_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['writer_user_count_category'] = pd.cut(df['writer_user_count'], bins=bins, labels=labels)
    return df


@feature(42, 'user_genre_count')
def user_genre_count(origin, df):
    user_genre_count = origin[['user', 'genre']].groupby('user')['genre'].agg('count').to_dict()
    df['user_genre_count'] = df['user'].map(user_genre_count)
    df.user_genre_count.fillna(0, inplace=True)
    df.user_genre_count = np.log(df.user_genre_count).astype(np.float32)
    return df


@feature(43, 'genre_user_count')
def genre_user_count(origin, df):
    genre_user_count = origin[['user', 'genre']].groupby('genre')['user'].count().reset_index().set_index('genre')['user']
    df['genre_user_count'] = df['genre'].map(genre_user_count)
    df.genre_user_count.fillna(0, inplace=True)
    df.genre_user_count = np.log(df.genre_user_count).astype(np.float32)
    return df


@feature(44, 'user_genre_count_category')
def user_genre_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['user_genre_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['user_genre_count_category'] = pd.cut(df['user_genre_count'], bins=bins, labels=labels)
    return df


@feature(45, 'genre_user_count_category')
def genre_user_count_category(origin, df):
    # 백분위수 기준으로 binning
    percentiles = [0, 25, 50, 75, 100]  # 백분위수 값을 기준으로 구간을 설정합니다.
    labels = [1,2,3,4]  # 각 구간에 대한 라벨을 설정합니다.
    bins = [df['genre_user_count'].quantile(p/100) for p in percentiles]  # 백분위수 값을 이용하여 구간을 설정합니다.
    bins[0] = -float('inf')  # 첫 번째 구간의 하한을 음의 무한대로 설정하여 0을 포함합니다.

    df['genre_user_count_category'] = pd.cut(df['genre_user_count'], bins=bins, labels=labels)
    return df


@feature(46, 'user_genre_review_count')
def user_genre_review_count(origin, df):
    pass


@feature(47, 'user_most_review_genre')
def user_most_review_genre(origin, df):
    pass


@feature(48, 'user_least_review_genre')
def user_least_review_genre(origin, df):
    pass


@feature(49, 'user_director_count')
def user_director_count(origin, df):
    # user_director_count = df[['user', 'director']].groupby('user')['director'].agg('count').to_dict()
    # df['user_director_count'] = df['user'].map(user_director_count)
    pass


@feature(50, 'director_user_count')
def director_user_count(origin, df):
    # director_user_count = df[['user', 'director']].groupby('director')['user'].agg('count').to_dict()
    # df['director_user_count'] = df['director'].map(director_user_count)
    pass


@feature(51, 'user_director_count_category')
def user_director_count_category(origin, df):
    pass


@feature(52, 'director_user_count_category')
def director_user_count_category(origin, df):
    pass
