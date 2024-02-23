import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, num_featrues, cat_features_size, emb_dim, mlp_dims, cross_feature):
        super().__init__()
        self.wide = Wide(num_featrues, cat_features_size, emb_dim, cross_feature)
        self.deep = Deep(num_featrues, cat_features_size, emb_dim, mlp_dims)
        self.bias = nn.Parameter(torch.rand_like(torch.zeros(1)))

    def forward(self, x):
        return torch.sigmoid(self.wide(x) + self.deep(x) + self.bias)


class Wide(nn.Module):
    def __init__(self, num_features, cat_features_size: dict, emb_dim, cross_feature):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_features = num_features
        self.cat_features_size = cat_features_size
        self.cross_feature = cross_feature
        
        self.embedding = nn.ModuleList([
            nn.Embedding(size, self.emb_dim) for _, size in cat_features_size.items()
        ])
        self.cross_product_features_size = len(self.cat_features_size)
        self.input_dim = len(self.num_features) + \
            len(self.cat_features_size) * self.emb_dim + \
            (self.cross_product_features_size * (self.cross_product_features_size-1))//2 * self.emb_dim
        
        self.fc = nn.Linear(self.input_dim, 1)
        self.size_check = True
    
    def forward(self, x: torch.Tensor):
        cat_feature_offset = len(self.num_features)
        num_features = x[:, :cat_feature_offset]
        cat_features_emb = [embedding(x[:, cat_feature_offset + i].long()) for i, embedding in enumerate(self.embedding)]
        cross_product_features = self._cross_product_features(cat_features_emb)
        x_flatten = torch.cat([num_features] + cat_features_emb + cross_product_features, dim=-1)

        if self.size_check:
            assert x_flatten.size() == (x.size(0), self.input_dim), f"Wide Component의 입력이 예상과 다릅니다.{x_flatten.size()} != {(x.size(0), self.input_dim)}"
            self.size_check = False
        
        return self.fc(x_flatten)

    def _cross_product_features(self, cat_features_emb):
        cross_features = []
        if not self.cross_feature:
            return cross_features
        
        for l in range(len(cat_features_emb)):
            left = cat_features_emb[l]
            for r in range(l + 1, len(cat_features_emb)):
                right = cat_features_emb[r]
                cross_features.append(left * right) # element-wise product
        
        return cross_features


class Deep(nn.Module):
    def __init__(self, num_features: list, cat_features_size: dict, emb_dim: int, mlp_dims: list):
        super().__init__()
        self.num_features = num_features
        self.cat_features_size = cat_features_size
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims

        self.embedding = nn.ModuleList([
            nn.Embedding(size, self.emb_dim) for _, size in cat_features_size.items()
        ])
        self.input_dim = len(self.num_features) + \
            len(self.cat_features_size) * self.emb_dim
        
        self.mlp = nn.ModuleList()
        for i, dim in enumerate(mlp_dims):
            if i == 0:
                self.mlp.append(nn.Linear(self.input_dim, dim))
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Dropout(0.2))
                prev_output_dim = dim
                continue
            self.mlp.append(nn.Linear(prev_output_dim, dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(0.2))
            prev_output_dim = dim
        self.mlp.append(nn.Linear(prev_output_dim, 1))
        self.mlp = nn.Sequential(*self.mlp)

        self.size_check = True

    def forward(self, x: torch.Tensor):
        cat_feature_offset = len(self.num_features)
        num_features = x[:, :cat_feature_offset]
        cat_features_emb = [embedding(x[:, cat_feature_offset + i].long()) for i, embedding in enumerate(self.embedding)]
        x_flatten = torch.cat([num_features] + cat_features_emb, dim=-1)

        if self.size_check:
            assert x_flatten.size() == (x.size(0), self.input_dim), f"Deep Component의 입력이 예상과 다릅니다.{x_flatten.size()} != {(x.size(0), self.input_dim)}"
            self.size_check = False
        
        return self.mlp(x_flatten)