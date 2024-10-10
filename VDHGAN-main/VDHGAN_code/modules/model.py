import torch
from torch import nn
import torch.nn.functional as f
from dgl.sampling import RandomWalkNeighborSampler
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import numpy as np


from mlp_readout import MLPReadout

from dgl.nn.pytorch import edge_softmax, GATConv



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(torch.nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_metapath, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []

        for i, block in enumerate(block_list):
            semantic_embeddings.append(
                self.gat_layers[i](block, h_list[i]).flatten(1)
            )
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_metapath,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

        output_dim = 64
        self.MPL_layer = MLPReadout(output_dim, 2)
        self.sigmoid = nn.Sigmoid()





    def forward(self, batch, cuda="cuda:0"):
        block_list, h_list, edge_types = batch.generate_meta_graph()
        g = [block.to(cuda) for block in block_list]
        h = [h.to(cuda) for h in h_list]
        for gnn in self.layers:
            h = gnn(g, h)

        h = self.predict(h)
        h = batch.de_batchify_graphs(h)
        outputs = h

        outputs = self.MPL_layer(outputs.sum(dim=1))
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs






if __name__=='__main__':

    def tally_param(model):
        total = 0
        for param in model.parameters():
            total += param.data.nelement()
        return total

    model = DevignModel(input_dim=100, output_dim=100,  #output_dim=100
                                num_steps=8, max_edge_types=5)
    model_2 = HAN(
        num_metapath=12,
        in_size=100,
        hidden_size=8,
        out_size=64,
        num_heads=[8],
        dropout=0.6,
    ).to("cuda:0")

    print(tally_param(model))
    print(tally_param(model_2))

    print(model)

