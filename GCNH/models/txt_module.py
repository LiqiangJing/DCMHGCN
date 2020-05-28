import torch
from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule
from torch.nn import Parameter
import math
from config import opt
LAYER1_NODE = 8192
LAYER1_NODE_GCN = 1024
label_num = 24
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
        self.conv1 = nn.Conv2d(1, 8192, kernel_size=(y_dim, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()
        
        self.feature = nn.Conv2d(8192, opt.Y_fea_nums, kernel_size=1, stride=(1, 1))
        self.relu2 = nn.ReLU()
        
        self.hashcode = nn.Conv2d(opt.Y_fea_nums, bit, kernel_size=1, stride=(1, 1))
        self.tanh1 = nn.Tanh()
        self.label = nn.Conv2d(opt.Y_fea_nums, label_num, kernel_size=1, stride=(1, 1))
        self.sigmoid1 = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.feature(x)
        x1 = self.relu2(x)
        
        x2 = self.hashcode(x1)
        x2 = self.tanh1(x2)
        
        x3 = self.label(x1)
        x3 = self.sigmoid1(x3)
        
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        x3 = x3.squeeze()
        return x1,x2,x3


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


class HashModule(BasicModule):
    def __init__(self, bit):
        super(HashModule, self).__init__()
        self.module_name = 'hash_model'

        self.gc1 = GraphConvolution(1024, LAYER1_NODE_GCN)
        self.relu1 = nn.ReLU()
        self.gc2 = GraphConvolution(LAYER1_NODE_GCN, opt.bit)
        self.tanh2 = nn.Tanh()
        
        self.conv3 = nn.Conv2d(1, label_num, kernel_size = (LAYER1_NODE_GCN, 1), stride=(1, 1))
        self.sigmoid3 = nn.Sigmoid()
    def forward(self,  inp, A):
        adj = gen_adj(A).detach()
        x = self.gc1(inp, adj)
        x = self.relu1(x)
        #hash
        x1 = self.gc2(x, adj)
        x1 = self.tanh2(x1)
        x1 = x1.squeeze()
        #lable
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.conv3(x)
        x = self.sigmoid3(x)
        
        x2 = x.squeeze()
        return x1,x2

def gen_adj(A):
    A = A + 1
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
class LabModule(BasicModule):
    def __init__(self, label_num):
        super(LabModule, self).__init__()
        self.module_name = 'lab_model'

        self.conv1 = nn.Conv2d(1, 512, kernel_size=(label_num, 1), stride = (1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 512, kernel_size = 1, stride = (1, 1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(512, opt.bit, kernel_size=1, stride = (1, 1))
        self.tanh3 = nn.Tanh()
        self.conv4 = nn.Conv2d(512, label_num, kernel_size=1, stride = (1, 1))
        self.sigmoid4 = nn.Sigmoid()
    def forward(self, input_label):
        x = self.conv1(input_label)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x1 = self.conv3(x)
        x1 = self.tanh3(x1)

        x2 = self.conv4(x)
        x2 = self.sigmoid4(x2)
        return x1.squeeze(), x2.squeeze()



if __name__ == '__main__':
    #A = gen_A(1386,0.1)
    data = torch.rand(23,1386)
    data = data.unsqueeze(1).unsqueeze(-1).type(torch.float)
    print(data.shape)
    model = TxtModule(1386,16)
    for i in range(1):
        X = model(data)

