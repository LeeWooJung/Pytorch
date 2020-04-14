import os
import argparse

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from utils import EarlyStop

parser = argparse.ArgumentParser()
parser.add_argument('--hiddenDims', type=list, default=[28*28,128,64,12,3], help='dimension of hidden layers, last value is the output dimensions of autoencoder output')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Number of batch sizz')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--window_size', type=int, default=10, help='window size')

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]) # since MNSIT has 1 channel
])

batch_size = 128

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def makeSeq(inDims, outDims):
	return nn.Sequential(nn.Linear(inDims,outDims), nn.ReLU(True))

class autoencoder(nn.Module):
	def __init__(self, hDims):
		super(autoencoder, self).__init__()

		self.hDims = hDims
		self.encoDims = hDims[:-1]
		ENC = [makeSeq(inDims, outDims) for inDims, outDims in zip(self.encoDims, self.encoDims[1:])]
		ENC.append(nn.Sequential(nn.Linear(self.hDims[-2],self.hDims[-1])))
		self.encoder = nn.Sequential(*ENC)

		self.decoDims = list(reversed(hDims))[:-1]
		DEC = [makeSeq(inDims, outDims) for inDims, outDims in zip(self.decoDims, self.decoDims[1:])]
		DEC.append(nn.Sequential(nn.Linear(self.hDims[1], self.hDims[0]), nn.Tanh()))
		self.decoder = nn.Sequential(*DEC)

	def forward(self,x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


args = parser.parse_args()
device = ("cuda" if torch.cuda.is_available() else "cpu")

hDims = args.hiddenDims
nLayer = len(hDims)
epochs = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
wd = args.weight_decay
ws = args.window_size

model = autoencoder(hDims)
model.to(device)

# Model tunning
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

# Train the model
es = [] # for early stopping
for epoch in range(1, epochs+1):
	total_loss = 0
	for data in dataloader:
		img, _ = data
		img = img.view(img.size(0),-1)
		img = Variable(img).to(device)

		output = model(img)
		loss = criterion(output, img)
		total_loss += loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	avg_loss = total_loss/len(dataloader.dataset)
	if epoch == 1:
		firstLoss = avg_loss
	es.append(avg_loss)
	if len(es) > ws:
		del es[0]
	if EarlyStop(es, ws, firstLoss):
		print("The loss doesn't decrease during window size.....")
		print("STOP training....................................")
		break
	print("Epoch {:3d}: average loss {:.4f}".format(epoch, avg_loss))
	if epoch % 10 == 0:
		pic = to_img(output.cpu().data)
		save_image(pic, './mlp_img/image_{}.png'.format(epoch))

print("Succesfully finished")
torch.save(model.state_dict(), './sim_autoencoder.pth')
