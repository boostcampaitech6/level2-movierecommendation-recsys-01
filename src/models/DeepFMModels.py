import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims.values())) 

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        self.embedding = nn.Embedding(total_input_dim, embedding_dim)
        self.embedding_dim = len(input_dims) * embedding_dim

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

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)        
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y

    def mlp(self, x):
        embed_x = self.embedding(x)

        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        #fm component
        fm_y = self.fm(x).squeeze(1)

        #deep component
        mlp_y = self.mlp(x).squeeze(1)

        y = torch.sigmoid(fm_y + mlp_y).unsqueeze(-1)
        return y
