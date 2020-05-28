import torch
from torch import nn
from config import opt
from torch.nn import functional as F
from models.basic_module import BasicModule
lable_num = 24
class ImgModule(BasicModule):
    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.fc8 = nn.Linear(in_features=4096,out_features=opt.X_fea_nums)
        self.relu8 = nn.ReLU()
        #fc9
        self.fc9 = nn.Linear(in_features=opt.X_fea_nums,out_features=bit)
        self.tanh9 = nn.Tanh()
        #fc10
        self.classifier = nn.Linear(in_features=opt.X_fea_nums, out_features=lable_num)
        self.sigmoid10 = nn.Sigmoid()
        self.fc8.weight.data = torch.randn(opt.X_fea_nums, 4096) * 0.1
        self.fc9.weight.data = torch.randn(bit, opt.X_fea_nums) * 0.1
        self.classifier.weight.data = torch.randn(lable_num, opt.X_fea_nums) * 0.01
        
        self.fc8.bias.data = torch.randn(opt.X_fea_nums) * 0.1
        self.fc9.bias.data = torch.randn(bit) * 0.1
        self.classifier.bias.data = torch.randn(lable_num) * 0.01
        
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        labnet = {}
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.fc8(x)
        x = self.relu8(x)
        labnet['feature'] = x
        
        x1 = self.fc9(x)
        x1 = self.tanh9(x1)
        labnet['hash'] = x1
        
        x = self.classifier(x)
        x = self.sigmoid10(x)
        labnet['label'] = x
        
        return labnet['feature'].squeeze(),labnet['hash'].squeeze(),labnet['label'].squeeze()


if __name__ == '__main__':
    path = '/home/jingliqiang/PycharmProjects/DCMH_pytorch/data/imagenet-vgg-f.mat'
    import scipy.io as scio
    data = scio.loadmat(path)
    print(data['normalization'][0][0])

