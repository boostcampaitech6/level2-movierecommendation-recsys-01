# Movie Recommendation
- Top-10 영화 추천 대회
- 매트릭: Recall@10
- 데이터: MovieLens 데이터를 implicit feedback으로 변형
	- test 데이터는 public: private=50: 50 으로 분할
- 실험관리: [실험관리](https://petite-giant-ce3.notion.site/9454685597b241d087b4fc0daae49147?pvs=25)
- 피처관리: [피처관리](https://petite-giant-ce3.notion.site/Feature-Engineering-3b25a82baffe4bc992ae3ad9722bade8?pvs=4)

# directory Hierarchy
```
.
├── README.md
├── ae-dataset.ipynb
├── ensemble.py
├── fm-test.ipynb
├── outputs
│   ├── datasets
│   ├── models
│   └── submissions
├── requirements.txt
├── run_inference.py
├── run_train.py
├── src
│   ├── __init__.py
│   ├── configs
│   ├── data
│   ├── ensembles.py
│   ├── inference
│   ├── loss.py
│   ├── metrics.py
│   ├── models
│   ├── train
│   └── utils.py
└── test.py

10 directories, 13 files
```

# Implemented

## Models
- FM/DeepFM
- AE/DAE/VAE/RecVAE

## Features
- ...
- ...

# How To Run

prerequisites
- conda
- git
- [torch](https://pytorch.org/#:~:text=Aid%20to%20Ukraine.-,INSTALL,-PYTORCH)

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
vi src/configs/inference_config.yaml
# inference
python run_inference.py
```
