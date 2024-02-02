# Movie Recommendation
- Top-10 영화 추천 대회
- 매트릭: Recall@10
- 데이터: MovieLens 데이터를 implicit feedback으로 변형
	- test 데이터는 public: private=50: 50 으로 분할
- 실험관리: [[ link ]]
- 데이터 관리: [[ dvc remote ]]

# directory Hierarchy
```
.
├── README.md
├── ensemble.py
├── requirements.txt
├── run_inference.py
├── run_train.py
└── src
    ├── configs
    ├── data
    ├── ensembles.py
    ├── loss.py
    ├── metrics.py
    ├── models
    ├── trainer.py
    └── utils.py

4 directories, 10 files
```

# Implemented

## Models
- DeepFM
- ...

## Features
- ...
- ...

# How To Run

prerequisites
- conda
- git
- torch [[ official homepage ]]

1. settings
```
# create virtual env
conda create -n movierec python==3.10
# activate virtual env
conda activate movierec
# install pytorch (CUDA 11.4)
pip install .....

# clone repository
git clone HERE
# change dirs
cd HERE
# install python libraries
pip install -r requirements
```

2. train
```
# configuration (vim, vscode, ...)
vi src/configs/train_config.yaml
# train model
python run_train.py
```

3. inference
```
# inference
vi src/configs/infer_config.yaml
# inference
python run_inference.py
```
