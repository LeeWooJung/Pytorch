import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchsummary import summary
from torch.utils.data import DataLoader

import random
import numpy as np
from tqdm import tqdm

# params
seed = 1234
batch_size = 64
lr = 0.1 # 50% of # of epochs: 0.01, 75% of # of epochs: 0.001
drop_prob = 0.2
epochs = 30
weight_decay = 1e-4
momentum = 0.9
nLables = 10
clip = 1.0
inDim = 3
theta = 0.5
growth_rate = 12
lowest_loss = float('inf')
filename = 'densnet.pt'

# fix seed to be reproducible
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Composition(nn.Module):
    def __init__(self, inDim, outDim, kernel_size, stride, padding, bias=False):
        super(Composition, self).__init__()
        self.bn = nn.BatchNorm2d(inDim)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(inDim, outDim, kernel_size = kernel_size,
                              stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

class BottleNeck(nn.Module):
    def __init__(self, inDim, k, drop_rate = 0.2): # k: growth rate
        super(BottleNeck, self).__init__()

        self.drop_rate = drop_rate
        self.DropOut = nn.Dropout(drop_rate, inplace=True)   
        self.conv1x1 = Composition(inDim = inDim, outDim = 4 * k, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv3x3 = Composition(inDim = 4 * k, outDim = k, kernel_size = 3, stride = 1, padding = 1, bias = False)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        if self.drop_rate > 0:
            out = self.DropOut(out)
        out = torch.cat((x, out), 1)

        return out, out.shape[1]

class DenseBlock(nn.Module):
    def __init__(self, inDim, nLayers, k, drop_rate):
        super(DenseBlock, self).__init__()
        
        self.growth_rate = k
        self.inDim = inDim
        self.nLayers = nLayers
        self.drop_rate = drop_rate

    def forward(self, x):
        grew_dim = self.inDim
        for _ in range(self.nLayers):
            bottleneck = BottleNeck(grew_dim, self.growth_rate, drop_rate=self.drop_rate)
            x, grew_dim = bottleneck(x)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, inDim, theta):
        super(TransitionBlock, self).__init__()
        self.composition = Composition(inDim = inDim, outDim = int(inDim * theta),
                                       kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.composition(x)
        out = self.avgpool(out)
        return out

class Densenet(nn.Module):
    def __init__(self, inDim, nLayers, nLabels, growth_rate, theta, filename, drop_rate = 0.2):
        super(Densenet, self).__init__()
        self.nLayers = nLayers
        self.nblockLayers = (nLayers - 4)//6
        self.outputsize = 0
        self.filename = filename

        self.firstLayer = nn.Conv2d(inDim, growth_rate*2, kernel_size=3, stride=1, padding=1, bias = True)

        self.denseblock1 = DenseBlock(inDim= growth_rate*2, nLayers=self.nblockLayers, k= growth_rate, drop_rate = drop_rate)
        self.outputsize = growth_rate * (2 + self.nblockLayers)
        self.transition1 = TransitionBlock(inDim = self.outputsize, theta=theta)

        self.denseblock2 = DenseBlock(inDim = int(self.outputsize * theta), nLayers=self.nblockLayers, k= growth_rate, drop_rate = drop_rate)
        self.outputsize = int(self.outputsize * theta) + growth_rate * self.nblockLayers
        self.transition2 = TransitionBlock(inDim = self.outputsize, theta=theta)

        self.denseblock3 = DenseBlock(inDim = int(self.outputsize * theta), nLayers = self.nblockLayers, k= growth_rate, drop_rate = drop_rate)
        self.outputsize = int(self.outputsize * theta) + growth_rate * self.nblockLayers

        self.lastLayer = nn.Linear(self.outputsize, nLabels)
        
    def forward(self, x):
        out = self.firstLayer(x)
        out = self.transition1(self.denseblock1(out))
        out = self.transition2(self.denseblock2(out))
        out = self.denseblock3(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.lastLayer(out)
        return out

def save_checkpoint(epoch, model, optimizer, schedule, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': schedule.state_dict()
    }
    torch.save(state, filename)

model = Densenet(inDim, nLayers=100, nLabels=10, growth_rate = growth_rate, theta = theta, filename=filename, drop_rate = drop_prob)

print("Summary of the model")
print(summary(model, (3, 224, 224)))


train = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform = transforms.ToTensor())
test = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform = transforms.ToTensor())

trMean = [np.mean(x.numpy(), axis = (1,2)) for x,_ in train]
trStd = [np.std(x.numpy(), axis = (1,2)) for x,_ in train]

trMean_RGB = [np.mean([m[index] for m in trMean]) for index in range(0,3)]
trStd_RGB = [np.mean([s[index] for s in trStd]) for index in range(0,3)]

trTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([trMean_RGB[0], trMean_RGB[1], trMean_RGB[2]],
    [trStd_RGB[0], trStd_RGB[1], trStd_RGB[2]]),
    transforms.RandomHorizontalFlip(),
])

tsTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([trMean_RGB[0], trMean_RGB[1], trMean_RGB[2]],
    [trStd_RGB[0], trStd_RGB[1], trStd_RGB[2]]),
])

train.transform = trTransform
test.transform = tsTransform
trloader = DataLoader(train, batch_size = batch_size, shuffle = True)
tsloader = DataLoader(test, batch_size = batch_size, shuffle = False)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum=momentum)
lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)])

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

        optimizer.step()
        lr_schedule.step()
        pbar.set_postfix(loss = loss.item())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(trloader)

    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        save_checkpoint(epochs, model, optimizer, lr_schedule, model.filename)

checkpoint = torch.load(model.filename)
model.load_state_dict(checkpoint['State_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
lr_schedule.load_state_dict(checkpoint['scheduler'])

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for i, (x,y) in enumerate(tsloader):
        x = x.to(device)
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print("Total test accuracy: {0:.2f".format(correct/total))