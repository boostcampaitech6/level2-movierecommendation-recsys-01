##### Table of Contents  
1. [Movie Recommendation](#프로젝트-개요)  
2. [프로젝트 구조](#프로젝트-구조)
3. [구현된 모델 목록](#구현된-모델-목록)
4. [기술 스택](#기술-스택)
5. [How To Run](#실행-방법)

---

# 프로젝트 개요
- 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 Top-10 영화를 예측하는 프로젝트
- 평가 지표:
  
  <img width="389" alt="image" src="https://github.com/boostcampaitech6/level2-movierecommendation-recsys-01/assets/95452963/248815c4-11dd-447f-8177-ed3ce90bd252">

- 데이터
  - MovieLens 데이터를 implicit feedback으로 변형
  - test 데이터는 public: private=50: 50 으로 분할
- [랩업리포트](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-01/blob/main/docs/%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%92%E1%85%AA%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%A5%E1%86%AB_Recsys_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(01%E1%84%8C%E1%85%A9).pdf)

# 프로젝트 구조
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

# 구현된 모델 목록

Collaborative Filtering
- [Matrix Factorization](https://ieeexplore.ieee.org/document/5197422)
- AutoEncoder-based models [AutoRec](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) [MultiVAE](https://arxiv.org/abs/1802.05814) [RecVAE](https://arxiv.org/abs/1912.11160)

Sequential
- [GRU4Rec](https://arxiv.org/abs/1511.06939)

Context-Aware
- [FM](https://ieeexplore.ieee.org/document/5694074)
- [Wide and Deep](https://arxiv.org/abs/1606.07792)
- [DeepFM](https://arxiv.org/abs/1703.04247)


# 기술 스택

![](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white)
![](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=black)
![](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white)

# 팀원 소개

| 팀원   | 역할 및 담당                      |
|--------|----------------------------------|
| [서동은](https://github.com/) | • EDA, ML modeling <br>• BPR-MF, kNN/ALS, Bert4rec |
| [신상우](https://github.com/sangwoonoel) | • 베이스라인 코드 작성 <br>• context-aware models(DeepFM), AE 기반 모델 구현 및 실험 |
| [이주연](https://github.com/twndus) | • 베이스라인 코드 구성하기 <br>• context-aware models(FM), AE 기반 모델 구현 및 실험 |
| [이현규](https://github.com/) | • EDA, ML modeling<br>• RecVAE, Rule-based, Bert4rec 사용|
| [이현주](https://github.com/uhhyunjoo) | • ML modeling, Hyper parameter tuning<br>• GRU4Rec 기반 모델링 |
| [조성홍](https://github.com/GangBean) | • 베이스라인 코드 작성<br>• context-aware models(FM,DeepFM,WDN) 기반 모델 구현 및 실험 |
