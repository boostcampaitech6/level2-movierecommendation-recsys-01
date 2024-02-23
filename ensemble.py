# ensemble.py
import pandas as pd

class Ensemble:
    def __init__(self, data_paths: list[str], type: str):
        self.inputs: list[pd.DataFrame] = self.read_files(data_paths)
        self.type: str = type
        self.output: pd.DataFrame = None

    def read_files(self, data_paths: list[str]):
        return [pd.read_csv(path) for path in data_paths]
    
    def make_result_file(self, output_path: str):
        self._ensemble() # type을 토대로 앙상블 수행
        self.output.to_csv(output_path) # 출력 파일 저장

    def _ensemble(self):
        if self.type == 'voting':
            self._voting_and_ranking()
        elif self.type == 'ranking':
            self._ranking()

    def _voting_and_ranking(self): # 유저별 아이템별 카운트
        min_vote_count = len(self.inputs)//2 + 1 # 50% 초과인 경우만 수용
        user_item_counts = self._user_item_count() # 유저별 아이템의 건수
            
    def _ranking(self, user_item_counts: dict, n: int=10):
        '''
        user_item_counts: {유저ID: {아이템ID: 각 모델에서 뽑힌 횟수}}
        n: 뽑으려는 아이템의 갯수
        '''
        output: dict = None
        output = {user: self._top_n_items(counts, n) for user, counts in user_item_counts.items()} # 유저별 top n 아이템 리스트 dict
        output_df = pd.DataFrame([(user, item) for user, items in output.items() for item in items], columns=['user', 'item'])
        self.output = output_df
    
    def _top_n_items(self, counts: dict, n: int=10):
        '''
        counts: {아이템ID: 각 모델에서 뽑힌 횟수}
        n: 뽑으려는 아이템의 갯수
        '''
        return [item[0] for item in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]
        
    def _user_item_count(self):
        '''
        전체 input을 취합해 유저별 아이템의 건수를 dict로 반환
        '''
        user_item_count = {}
        for input in self.inputs:
            for user, df in input.groupby('user'):
                tmp_count = df['item'].value_counts().to_dict()
                self._merge_dict(user_item_count[user], tmp_count)
        return user_item_count
    
    def _merge_dict(self, origin: dict, add: dict):
        '''
        기존 item-count dict에 새로운 item-count dict를 병합
        '''
        for item, count in add.items():
            if item in origin:
                origin[item] += count
                continue

            origin[item] = count

    def _ranking(self):
        pass