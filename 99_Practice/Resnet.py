import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchsummary import summary

import random
import numpy as np
from tqdm import tqdm

seed = 1234
batch_size = 256
lr = 0.001
clip = 1.0
epochs = 10
inDim = 3
nLabels = 10
lowest_loss = 1000

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Building Block
class BuildingBlock(nn.Module):
    expansion = 1
    def __init__(self, inChannel, outChannel, stride=1):
        super(BuildingBlock, self).__init__()
        
        self.residual = nn.Sequential(
            # BatchNorm2d에 Bias가 있으므로 Conv2d에는 Bias 제거
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel* BuildingBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannel*BuildingBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        # if stride != 1 then shortcut output dim != residual output dim
        if stride != 1 or inChannel != outChannel * BuildingBlock.expansion:
            # do 1x1 convolution
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel*BuildingBlock.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outChannel*BuildingBlock.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.relu(out)

        return out

# BottleNeck Block
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inChannel, outChannel, stride=1):
        super(BottleNeck, self).__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel*BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outChannel*BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inChannel != outChannel * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel*BottleNeck.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outChannel*BottleNeck.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.relu(out)

        return out

# Resnet
class ResNet(nn.Module):
    def __init__(self, block, filename, nBlocks, nLabels = 10, initialize = True):
        super(ResNet, self).__init__()
        
        self.inChannel = 64
        self.filename = filename
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inChannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self.multipleBlocks(block, 64, nBlocks[0], 1)
        self.conv3_x = self.multipleBlocks(block, 128, nBlocks[1], 2)
        self.conv4_x = self.multipleBlocks(block, 256, nBlocks[2], 2)
        self.conv5_x = self.multipleBlocks(block, 512, nBlocks[3], 2)

        self.avgerage_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, nLabels)

        if initialize:
            self.initialize_weight()

    def multipleBlocks(self, block, outChannel, nBlock, stride):
        strides = [stride] + [1] * (nBlock-1)
        layers = []
        for s in strides:
            layers.append(block(self.inChannel, outChannel, s))
            self.inChannel = outChannel * block.expansion

        return nn.Sequential(*layers)

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgerage_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)

def resnet18():
    filename = 'resnet18.pt'
    nBlocks = [2,2,2,2]
    return ResNet(BuildingBlock, filename, nBlocks)

def resnet34():
    filename = 'resnet34.pt'
    nBlocks = [3,4,6,3]
    return ResNet(BuildingBlock, filename, nBlocks)

def resnet50():
    filename = 'resnet50.pt'
    nBlocks = [3,4,6,3]
    return ResNet(BottleNeck, filename, nBlocks)

def resnet101():
    filename = 'resnet101.pt'
    nBlocks = [3,4,23,3]
    return ResNet(BottleNeck, filename, nBlocks)

def resnet152():
    filename = 'resent152.pt'
    nBlocks = [3,8,36,3]
    return ResNet(BottleNeck, filename, nBlocks)

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform= transforms.ToTensor())
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform= transforms.ToTensor())

trMean = [np.mean(x.numpy(), axis = (1,2)) for x, _ in train]
trStd = [np.std(x.numpy(), axis = (1,2)) for x, _ in train]

trMean_RGB = [np.mean([m[index] for m in trMean]) for index in range(0,3)]
trStd_RGB = [np.mean([s[index] for s in trStd]) for index in range(0,3)]

trTransformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([trMean_RGB[0], trMean_RGB[1], trMean_RGB[2]],
                         [trStd_RGB[0], trStd_RGB[1], trStd_RGB[2]]),
    transforms.RandomHorizontalFlip(),
])

tsTransformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([trMean_RGB[0], trMean_RGB[1], trMean_RGB[2]],
                         [trStd_RGB[0], trStd_RGB[1], trStd_RGB[2]])
])

train.transform = trTransformation
test.transform = tsTransformation
trloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
tsloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

model = resnet18().to(device)
x = torch.randn(3,3,224,224).to(device)
output = model(x)
print(output.size())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

for epoch in range(1, epochs+1):
    
    epoch_loss = 0.0
    model.train()
    pbar = tqdm(trloader)
    pbar.set_description("[(Train) Epoch {}]".format(epoch))

    for i, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x).to(device)

        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        pbar.set_postfix(loss = loss.item())
        epoch_loss += loss.item()

    epcoh_loss = epoch_loss / len(trloader)

    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        save_checkpoint(epochs, model, optimizer, model.filename)

checkpoint = torch.load(model.filename)
model.load_state_dict(checkpoint['State_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for i, (x, y) in enumerate(tsloader):
        x = x.to(device)
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print("Total test accuracy: {0:.2f}".format(correct/total))