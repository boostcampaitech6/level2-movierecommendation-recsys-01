import torch
import torch.nn as nn
import numpy as np

class DeepFM(nn.Module):
    def __init__(self, num_features:list, cat_features_size: dict, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = len(num_features) + int(sum(cat_features_size.values()))

        self.num_features = num_features
        self.cat_features_size = cat_features_size

        self.bias = nn.Parameter(torch.zeros((1,)))
        # self.fc = nn.Embedding(total_input_dim, 1)
        self.fc = nn.Parameter(torch.rand(total_input_dim, 1))

        self.cat_embedding = nn.Embedding(total_input_dim, embedding_dim)
        self.num_embedding = nn.Linear(len(num_features), embedding_dim)
        self.embedding_dim = len(cat_features_size) * embedding_dim + bool(len(self.num_features))

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) 
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x, embed_x):
        # x : (batch_size, total_num_input)
        fm_y = self.bias + torch.sum(torch.matmul(x, self.fc), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=-1, keepdim=True)
        return fm_y

    def mlp(self, x):
        inputs = x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        cat_offset = len(self.num_features)
        cat_embed_x = self.cat_embedding(x[:,cat_offset:].long())
        cat_embed_x = cat_embed_x.view(cat_embed_x.size(0), -1)
        if cat_offset > 0:
            num_embed_x = self.num_embedding(x[:,:cat_offset]) # (batch_size, emb_dim) 그 결과 
            embed_x = torch.cat([num_embed_x, cat_embed_x], dim=-1)
        else:
            embed_x = cat_embed_x

        # split numeric and categorical features value
        num_x = x[:, :cat_offset]
        cat_x = x[:, cat_offset:].long()
        # one-hot transformation
        cat_x = cat_x + cat_x.new_tensor([0, *np.cumsum(self.cat_features_size.values())[:-1]])
        cat_x = torch.zeros(cat_x.size(0), sum(self.cat_features_size.values()), device=cat_x.device).scatter_(1, cat_x, 1.)
        # add offsets to categorical features value
        if cat_offset > 0:
            x_fm = torch.concat([num_x, cat_x], axis=1)
        else:
            x_fm = cat_x
        
        #fm component
        fm_y = self.fm(x_fm, embed_x).unsqueeze(1) # 128, 1

        #deep component
        mlp_y = self.mlp(embed_x) # *128 , 1

        y = torch.sigmoid(fm_y + mlp_y) # 128, 1
        # print(fm_y.size())
        # print(mlp_y.size())
        # print(y.size())
        return y
