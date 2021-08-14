from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math
from MNIST_shift import shift_image_v2


#    -----------------Helper Functions-----------------
def log_gaussian(x,mean,var):
    std = math.sqrt(var)
    return -math.log(std*math.sqrt(2*math.pi)+1e-4) - 0.5*((x-mean)/(std+1e-4))**2

def log_gausian_torch(x,mean,var):
    std = torch.sqrt(var)
    return -torch.log(std*math.sqrt(2*math.pi)+1e-4) - 0.5*((x-mean)/(std+1e-4))**2

def accuracy(y_true, y_pred):
    #y_true and y_pred is assumed to be numpy array.
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_pred.reshape(-1).astype(int)

    correct = 0
    for i in range(0,y_true.shape[0]):
        if y_true[i] == y_pred[i]:
            correct = correct + 1

    return correct/y_true.shape[0]

def shift_image(x,y,width_shift_val,height_shift_val):
    #x is assumed to be a tensor of shape (#batch_size, #channels, width, height) = (#batch size, 1, 28, 28)
    #y is assumed to be a tensor of shape (#batch_size)
    batch_size = x.size()[0]
    x = x.detach().cpu().numpy().reshape(batch_size,28,28)
    y = y.detach().cpu().numpy().reshape(batch_size)

    # import relevant library
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # create the class object
    datagen = ImageDataGenerator(width_shift_range=width_shift_val, height_shift_range=height_shift_val)
    # fit the generator
    datagen.fit(x.reshape(batch_size, 28, 28, 1))

    a = datagen.flow(x.reshape(batch_size, 28, 28, 1),y.reshape(batch_size, 1),batch_size=batch_size,shuffle=False)

    X, Y = next(iter(a))   
    X = torch.from_numpy(X).view(batch_size,1,28,28)
    Y = torch.from_numpy(Y).view(batch_size)

    return X,Y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#  ----------------------------------------------------

#  ----------------------------------------------------

    
parser = argparse.ArgumentParser(description='DeepCAMA MNIST Example')
parser.add_argument('--train', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_set = datasets.MNIST('./data/train/', train=True, download=True, transform=transforms.ToTensor())
train_data, val_data = torch.utils.data.random_split(train_set, [int(0.95*len(train_set)),int(0.05*len(train_set))], generator=torch.Generator().manual_seed(42))
test_data = datasets.MNIST('./data/train/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
val_lodaer = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)
# ------------------------------------------------------\


class DeepCAMABaseline(nn.Module):
    def __init__(self):
        super(DeepCAMABaseline, self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,126)
        self.fc4 = nn.Linear(126,512)
        self.fc5 = nn.Linear(512,28*28)
        self.d1 = nn.Dropout(p=0.25)
        self.d2 = nn.Dropout(p=0.5)



    def forward(self, x):
        a1 = self.d1(F.relu(self.fc1(x.view(-1,28*28))))
        a2 = self.d1(F.relu(self.fc2(a1)))
        a3 = self.d1(F.relu(self.fc3(a2)))
        a4 = self.d2(F.relu(self.fc4(a3)))
        a5 = self.fc5(a4)
        
        return a5


def train(epoch):
    train_loss = 0
    for batch_id,  (data,y) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device)
        model.train()
        optimizer.zero_grad()
        x_recon = model(data)
        loss = loss_function(x_recon,y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_id % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return

def pred(y):

    return 
    

model = DeepCAMABaseline().to(device)
optimizer = optim.Adam(model.parameters(), lr=(1e-4+1e-5)/2)
loss_function = nn.CrossEntropyLoss()

if __name__ == "__main__":
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
        torch.save(model.state_dict(), '/media/hsy/DeepCAMA/baseline.pt') 
        
    
    
    
    
    
