import numpy as np

import torch
import torch.nn as nn


class FMLayer(nn.Module):
    def __init__(self, input_dim, emb_dim):
        '''
        Parameter
            input_dim: Input dimension in sparse representation (in MovieLens)
            emb_dim: Factorization dimension
        '''
        super(FMLayer, self).__init__()
        self.v = nn.Parameter(
            torch.empty(input_dim, emb_dim)  
            , requires_grad = True
        )

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, x):
        '''
        Parameter
            x: Float tensor of size "(batch_size, input_dim)"
        '''
        square_of_sum = self.square(torch.matmul(x, self.v)) 
        sum_of_square = torch.matmul(self.square(x), self.square(self.v)) 

        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)

        
class FM(nn.Module):
    def __init__(self, num_features: list, cat_features_size: dict, emb_dim: int):
        '''
        Parameter
            input_dim: Input dimension in sparse representation (2652 in MovieLens-100k)
            factor_dim: Factorization dimension
        '''
        super(FM, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_features = num_features
        self.cat_features_size = cat_features_size
        self.input_dim = len(self.num_features) + sum(self.cat_features_size.values())
        
        self.con_bias = [nn.Parameter(torch.zeros((1,))) for _ in range(len(self.num_features))]
        self.cat_bias = nn.ModuleList([
            nn.Embedding(feature_size, 1) for _, feature_size in self.cat_features_size.items()])

        self.fm = FMLayer(self.input_dim, self.emb_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, FMLayer):
                nn.init.normal_(m.v, 0, 0.01)

    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, input_dim)"

        Return
            y: Float tensor of size "(batch_size)"
        '''
        # global bias
        # con_bias = [bias.unsqueeze(0).expand(x.size(0), -1).to(x.device) for bias in self.con_bias] # (batch_size, 1)
 
        cat_index_offset = len(self.num_features)
        cat_bias = [emb(x[:,cat_index_offset + idx]) for idx, emb in enumerate(self.cat_bias)] # u_bias, i_bias
        
        bias = torch.cat(cat_bias, axis=-1)
        bias_term = torch.sum(bias, axis=1)

        # split numeric and categorical features value
        num_x = x[:, :cat_index_offset]
        cat_x = x[:, cat_index_offset:]

        # one-hot transformation
        # add offsets to categorical features value
        cat_x = cat_x + cat_x.new_tensor([0, *np.cumsum(self.cat_features_size.values())[:-1]])
        cat_x = torch.zeros(cat_x.size(0), sum(self.cat_features_size.values()), device=cat_x.device).scatter_(1, cat_x, 1.)
        x_fm = torch.concat([num_x, cat_x], axis=1)

        # concat results
        y = bias_term + self.fm(x_fm)
        y = torch.sigmoid(y)

        return y.unsqueeze(-1)
