import torch
from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule
from torch.nn import Parameter
import math
LAYER1_NODE = 8192
import numpy as np
def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x


class GraphConvolution(BasicModule):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    y_dim: dimension of tags
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TxtGcnnet(BasicModule):
    def __init__(self,num_classes, bit, in_channel=1, t=0.4):
        super(TxtGcnnet, self).__init__()

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, bit)
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(1,bit,kernel_size=(num_classes,bit),stride=(1,1))
        _adj = gen_A(num_classes, t)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self,  inp):
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze()
        return x

def gen_A(num_classes, t, adj_file='/home/share/sunteng/jing/bishe/data/Adj.mat'):
    import scipy.io as so
    result = so.loadmat(adj_file)
    _adj = result['Adj']
    _nums = result['Num'].squeeze()
    _nums = _nums[:, np.newaxis]

    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

if __name__ == '__main__':
    A = gen_A(1386,0.1)
    data = torch.rand(23,1386)
    data = data.unsqueeze(-1)
    print(data.shape)
    model = TxtGcnnet(1386,16)
    for i in range(1):
        X = model(data)
        print(X.shape)

