import torch
import torch.nn as nn
from torchsummary import summary

import random
import numpy as np

S = 7
B = 2
C = 20
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Convnet(nn.Module):
    def __init__(self, inDim, outDim, kernel, stride, padding):
        super(Convnet, self).__init__()
        self.conv = nn.Conv2d(inDim, outDim, kernel, stride, padding, bias = False)
        self.batchnorm = nn.BatchNorm2d(outDim)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lrelu(self.batchnorm(self.conv(x)))

class Yolo(nn.Module):
    def __init__(self, conv, structure, S = 7, B = 2, C = 20):
        super(Yolo, self).__init__()
        
        self.networks = self.make_layers(conv, structure)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, (S*S*(B*5 + C)))
        )

    def make_layers(self, conv, structure):
        layers = []
        inDim = 3
        for s in structure:

            if type(s) == tuple:
                layers.append(conv(inDim, s[1], s[0], s[2], s[3]))
                inDim = s[1]
            elif type(s) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.networks(x)
        out = self.fc(torch.flatten(out, 1))
        return out

# Structure
# (Kernel_size, Output dim, Stride, Padding)
structure = [
    (7, 64, 2, 3),
    "Maxpool",
    (3, 192, 1, 1),
    "Maxpool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "Maxpool",
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "Maxpool",
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

model = Yolo(Convnet, structure)
print(summary(model, (3, 448, 448)))