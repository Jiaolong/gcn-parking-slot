import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)
        self.bn3 = nn.BatchNorm2d(input_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, input_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(input_dim*2, input_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(input_dim*3, output_dim, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x):
        """
        Inputs:
            x: (batch_size, input_dim, num_points)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x1 = x

        x = get_graph_feature(x1, k=self.k)  # (batch_size, input_dim, num_points) -> (batch_size, input_dim*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, input_dim*2, num_points, k) -> (batch_size, input_dim, num_points, k)
        x = self.conv2(x)                       # (batch_size, input_dim, num_points, k) -> (batch_size, input_dim, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, input_dim, num_points, k) -> (batch_size, input_dim, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, input_dim, num_points) -> (batch_size, input_dim*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, input_dim*2, num_points, k) -> (batch_size, input_dim, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, input_dim, num_points, k) -> (batch_size, input_dim, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, input_dim*3, num_points)
        
        #print('after concat: ', x.size())
        x = self.conv4(x)                       # (batch_size, input_dim*3, num_points) -> (batch_size, output_dim, num_points)
        #print('after conv4: ', x.size())
        return x
