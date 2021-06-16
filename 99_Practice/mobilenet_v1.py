import torch
import torch.nn as nn
from torchsummary import summary

import math
import random
import numpy as np

# params
seed = 1234
nLabels = 1000
alpha = 1.0 # width multiplier

# fix seed to be reproducible
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DepthWiseSeparable(nn.Module):
    def __init__(self, inDim, outDim, stride = 1, padding = 1):
        super(DepthWiseSeparable, self).__init__()
        
        self.DWsep = nn.Sequential(
            nn.Conv2d(inDim, inDim, kernel_size = 3, stride = stride, padding = padding, groups = inDim, bias = False),
            nn.BatchNorm2d(inDim),
            nn.ReLU(True),
            nn.Conv2d(inDim, outDim, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(outDim),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.DWsep(x)
        return out

class MobilenetV1(nn.Module):
    def __init__(self, block, inDims, outDims, strides, nLabels = 1000):
        super(MobilenetV1, self).__init__()
        
        self.FirstLayer = nn.Sequential(
            nn.Conv2d(3, inDims[0], kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(inDims[0]),
            nn.ReLU(True)
        )
        self.DSBlocks = self.make_layer(block, inDims, outDims, strides)
        self.avgPool = nn.AvgPool2d(7)
        self.LastLayer = nn.Linear(outDims[-1], nLabels)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def make_layer(self, block, inDims, outDims, strides):
        layers = []
        for index, _ in enumerate(inDims):
            layers.append(block(inDims[index], outDims[index], strides[index]))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.FirstLayer(x)
        out = self.DSBlocks(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.LastLayer(out)
        return out

strides = [1,2,1,2,1,2,1,1,1,1,1,2,1]
origin_inDims = [32,64,128,128,256,256,512,512,512,512,512,512,1024]
origin_outDims = [64,128,128,256,256,512,512,512,512,512,512,1024,1024]

inDims = [int(alpha * indim) for indim in origin_inDims]
outDims = [int(alpha * outdim) for outdim in origin_outDims]

model = MobilenetV1(DepthWiseSeparable, inDims, outDims, strides, nLabels)

print(summary(model, (3, 224, 224)))