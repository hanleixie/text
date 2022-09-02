# -*- coding: utf-8 -*-
# @Time     : 2022/2/17 9:02
# @Author   : xiehl
# @Software : PyCharm
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # 在模型参数更新时，A（邻接矩阵） 不更新
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks1 = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
        ))
        if edge_importance_weighting:
            self.edge_importance1 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks1
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks1)

        self.st_gcn_networks2 = nn.ModuleList((
            st_gcn(64, 128, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
        ))
        if edge_importance_weighting:
            self.edge_importance2 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks2
            ])
        else:
            self.edge_importance2 = [1] * len(self.st_gcn_networks2)
        self.st_gcn_networks3 = nn.ModuleList((
            st_gcn(128, 256, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance3 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks3
            ])
        else:
            self.edge_importance3 = [1] * len(self.st_gcn_networks3)

        # fcn for prediction
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.AvgPool3d((1, 1, 1), stride=(1, 1, 1))
        self.pool2 = nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1))
        self.pool3 = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))


    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks1, self.edge_importance1):
            x, _ = gcn(x, self.A * importance)
            x_pool1 = self.pool1(x)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks2, self.edge_importance2):
            x, _ = gcn(x, self.A * importance)
            x_pool2 = self.pool2(x)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks3, self.edge_importance3):
            x, _ = gcn(x, self.A * importance)
            x_pool3 = self.pool3(x)

        x = x_pool1 + x_pool2 + x_pool3
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        # print(x.shape)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return F.softmax(x, dim=1)
        # return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        # gcn模型，此模型完成空间图卷积操作
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        # tcn模型，此模型完成对在空间图卷积下的时间预测
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        # 是否使用残差结构
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.cam = CAM()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        x = self.relu(x)
        # add
        x = self.cam(x)
        x = self.relu(x)
        return x, A

    # def channel_wise_attention(self, feature_map):
    #
    #     _, H, W, C = feature_map.shape
    #     w_s = nn.init.orthogonal_(torch.empty(C, C))
    #     b_s = torch.zeros(C)
    #     org_mean = torch.mean(feature_map, dim=[1, 2], keepdim=True)
    #     transpose_feature_map = org_mean.permute(0, 3, 1, 2)
    #     channel_wise_attention_fm = torch.matmul(transpose_feature_map.reshape(-1, C), w_s) + b_s
    #     channel_wise_attention_fm = torch.sigmoid(channel_wise_attention_fm)
    #     attention = torch.cat([channel_wise_attention_fm] * (H * W), axis=1).reshape(-1, H, W, C)
    #     attended_fm = attention * feature_map
    #     return attended_fm


# class Channel_wise_attention(nn.Module):
#     """
#     Attention Network.
#     """
#
#     def __init__(self, feature_map, decoder_dim, K=512):
#         """
#         :param feature_map: feature map in level L
#         :param decoder_dim: size of decoder's RNN
#         """
#         super(Channel_wise_attention, self).__init__()
#         _, C, H, W = tuple([int(x) for x in feature_map])
#         self.W_c = nn.Parameter(torch.randn(1, K))
#         self.W_hc = nn.Parameter(torch.randn(K, decoder_dim))
#         self.W_i_hat = nn.Parameter(torch.randn(K, 1))
#         self.bc = nn.Parameter(torch.randn(K))
#         self.bi_hat = nn.Parameter(torch.randn(1))
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax(dim=0)  # softmax layer to calculate weights
#
#     def forward(self, feature_map, decoder_hidden):
#         """
#         Forward propagation.
#         :param feature_map: feature map in level L(batch_size, C, H, W)
#         :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
#         :return: alpha
#         """
#         V_map = feature_map.view(feature_map.shape[0], 2048, -1).mean(dim=2)
#         V_map = V_map.unsqueeze(2)  # (batch_size,C,1)
#         # print(feature_map.shape)
#         # print(V_map.shape)
#         # print("wc",self.W_c.shape)
#         # print("whc",self.W_hc.shape)
#         # print("decoder_hidden",decoder_hidden.shape)
#         # print("m1",torch.matmul(V_map,self.W_c).shape)
#         # print("m2",torch.matmul(decoder_hidden,self.W_hc).shape)
#         # print("bc",self.bc.shape)
#         att = self.tanh((torch.matmul(V_map, self.W_c) + self.bc) + (
#             torch.matmul(decoder_hidden, self.W_hc).unsqueeze(1)))  # (batch_size,C,K)
#         #         print("att",att.shape)
#         beta = self.softmax(torch.matmul(att, self.W_i_hat) + self.bi_hat)
#         beta = beta.unsqueeze(2)
#         # print("beta",beta.shape)
#         attention_weighted_encoding = torch.mul(feature_map, beta)
#
#         return attention_weighted_encoding, beta


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = F.softmax(energy, dim=-1)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)

        out = self.para_mu*out + x
        return out
