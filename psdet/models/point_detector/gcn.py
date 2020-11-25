import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
from .dgcnn import DGCNN

def MLP(channels: list, do_bn=True, drop_out=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

class EdgePredictor(nn.Module):
    "Edge connectivity predictor using MLPs"
    def __init__(self, cfg):
        super(EdgePredictor, self).__init__()
        self.encoder = MLP([cfg.input_dim] + cfg.layers + [1], drop_out=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, data_dict):
        x = data_dict['descriptors']
        b, c, n = x.shape
        inputs = torch.zeros(b, c * 2, n * n).cuda()
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                inputs[:, :, idx] = torch.cat([x[:, :, i], x[:, :, j]], dim=1)
        
        preds = torch.sigmoid(self.encoder(inputs))
        data_dict['edges_pred'] = preds
        return data_dict

class PointEncoder(nn.Module):
    """ Joint encoding of depth, intensity and location (x, y, z) using MLPs"""
    def __init__(self, input_dim, feature_dim, layers):
        super(PointEncoder, self).__init__()
        self.encoder = MLP([input_dim] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, pts):
        return self.encoder(pts)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super(AttentionalPropagation, self).__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_layers)])

    def forward(self, desc0, desc1=None):
        if desc1 is None:
             return self.forward_self_attention(desc0)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                src0, src1 = desc0, desc1
            else:  # cross attention
                src0, src1 = desc1, desc0
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0
    
    def forward_self_attention(self, x):
        for layer in self.layers:
            src = x
            delta = layer(x, src)
            x = x + delta
        return x

class GCNEncoder(nn.Module):
    def __init__(self, cfg):
        super(GCNEncoder, self).__init__()
        self.cfg = cfg
        self.feat_dim = cfg.point_encoder.output_dim

        self.point_encoder = PointEncoder(2, self.feat_dim, cfg.point_encoder.layers) 
        
        if cfg.type == 'GAT':
            self.gnn = AttentionalGNN(self.feat_dim, cfg.gnn.gat_layers)
        elif cfg.type == 'DGCNN':
            self.gnn = DGCNN(self.feat_dim, self.feat_dim, k=cfg.gnn.k)
        else:
            raise ValueError("Unknown GCNEncoder type {}".format(cfg.type))

        self.proj = nn.Conv1d(self.feat_dim, cfg.gnn.proj_dim, kernel_size=1, bias=True)

    def forward(self, data_dict):
        points = data_dict['points'][:,:,:2] # [B, N num_points, 2]

        points = points.permute(0, 2, 1) # [B, 2, num_points]

        x = self.point_encoder(points) # [B, desc_dim, num_points]

        desc = data_dict['descriptors'] # [B, desc_dim, num_points]

        x += desc
        
        # Multi-layer Transformer network.
        x = self.gnn(x)

        # MLP projection.
        x = self.proj(x)
                
        data_dict['descriptors'] = x
        return data_dict
