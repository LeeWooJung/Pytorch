import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import random
import numpy as np
from tqdm import tqdm

seed = 1234
batch_size = 128
lr = 0.001
drop_prob = 0.2
clip = 1.0
epochs = 10
inDim = 3
nLabels = 10
vgg_type = 'A'
lowest_loss = 1000
filename = 'vgg.pt'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class vgg(nn.Module):
    def __init__(self, layers, inDim, nLabels, drop_prob):
        super(vgg, self).__init__()

        self.layers = layers
        self.avgPool = nn.AdaptiveAvgPool2d(7)
        self.FCs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Linear(4096, nLabels)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.FCs(x)
        return x

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)

def vgg_layers(cfg, batch_norm = False):
    layers = []
    inchannels = 3
    for val in cfg:
        if val == 'M': # Max pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(inchannels, val, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(val), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            inchannels = val

    return nn.Sequential(*layers)

cfg = { 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }

layers = vgg_layers(cfg[vgg_type], batch_norm=True)
model = vgg(layers = layers, inDim = inDim, nLabels=nLabels, drop_prob=drop_prob).to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)

trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

for epoch in range(1, epochs+1):
    
    epoch_loss = 0.0
    model.train()
    pbar = tqdm(trainloader)
    pbar.set_description("[(Train) Epoch {}]".format(epoch))

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images).to(device)

        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        pbar.set_postfix(loss = loss.item())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(trainloader)

    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        save_checkpoint(epochs, model, optimizer, filename)

checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['State_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Total test accuracy: {0:.2f}".format(correct/total))