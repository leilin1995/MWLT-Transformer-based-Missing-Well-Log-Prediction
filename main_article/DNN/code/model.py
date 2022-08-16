"""
__author__ = 'linlei'
__project__:model
__time__:2022/7/6 19:33
__email__:"919711601@qq.com"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class DNN(nn.Module):
    def __init__(self, in_features=4, out_features=1, dropout=0.1):
        super().__init__()
        self.dropout =dropout
        self.fc1=nn.Sequential(nn.Linear(in_features=in_features, out_features=1024),
                               nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(in_features=1024, out_features=1024),
                               nn.ReLU())
        self.fc3=nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                               nn.ReLU())
        self.fc4=nn.Sequential(nn.Linear(in_features=512, out_features=1),
                               nn.Sigmoid() # normal
                               # nn.ReLU()  # nonormal
        )

    def forward(self,x):
        x=self.fc1(x)
        x=F.dropout(x,p=self.dropout)
        x=self.fc2(x)
        x=F.dropout(x,p=self.dropout)
        x=self.fc3(x)
        x=F.dropout(x,p=self.dropout)
        x=self.fc4(x)
        return x


def claculate_flop_param():
    model=DNN()
    data=torch.randn(640,4)
    flops, params = profile(model, inputs=(data,))
    with open("params_flops.txt","w") as f:
        f.write("DNN: params={:.3f}M, flops={:.3f}G\n".format(params / 1000 ** 2,flops / 1000 ** 3))
        f.close()

if __name__ == "__main__":
    claculate_flop_param()