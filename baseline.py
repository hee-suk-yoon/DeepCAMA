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
from util import *


    
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

train_data = datasets.MNIST('./data/train/', train=True, download=True, transform=transforms.ToTensor())
train_data_clean = list(train_data)
for idx, _ in enumerate(train_data_clean):
    L1 = list(train_data_clean[idx])
    L1.append(0)
    train_data_clean[idx] = tuple(L1)
    
#train_loader = torch.utils.data.DataLoader(train_data_clean, batch_size=args.batch_size, shuffle=True, **kwargs)

vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)

listofones = [1] * 10
split_list = [int(element * 0.1*len(train_data)) for element in listofones]
train_data_aug_split = torch.utils.data.random_split(train_data, split_list, generator=torch.Generator().manual_seed(42))
train_data_aug = []
for idx, _ in enumerate(train_data_aug_split):
    split_part = list(train_data_aug_split[idx])
    for idx2, _ in enumerate(split_part):
        #print(split_part[idx2][0].size())
        shift_fn = transforms.RandomAffine(degrees=0,translate=(0.0,vertical_shift_range[idx]))
        L1 = list(split_part[idx2])
        L1[0] = shift_fn(L1[0])
        L1.append(1)
        split_part[idx2] = tuple(L1)
        #train_data_aug_split[idx][idx2] = tuple(L1)
        #print(L1)
    train_data_aug = train_data_aug + split_part
    
#for idx, split_part in enumerate(train_data_aug_split):
#    #print(list(split_part)[0][0].size())
#    train_data_aug = train_data_aug + list(split_part)       

#for idx, _ in enumerate(train_data_aug_split):
#    #print(list(split_part)[0][0].size())
#    split_part = list(train_data_aug_split[idx])
#    train_data_aug = train_data_aug + split_part    

train_data_clean_aug = train_data_clean + train_data_aug
train_loader = torch.utils.data.DataLoader(train_data_clean_aug, batch_size=args.batch_size, shuffle=True, **kwargs)
print(len(train_data_aug))
print(len(train_data_clean_aug))

    












test_data = datasets.MNIST('./data/train/', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)
# ------------------------------------------------------\


class DeepCAMABaseline(nn.Module):
    def __init__(self):
        super(DeepCAMABaseline, self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,126)
        self.fc4 = nn.Linear(126,512)
        self.fc5 = nn.Linear(512, 10)
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
    for batch_id,  (data,y,clean) in enumerate(train_loader):
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

def pred(x):
    out = model(x.to(device))
    #y = y.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    label = np.argmax(out,axis = 1)
    return label
    

model = DeepCAMABaseline().to(device)
optimizer = optim.Adam(model.parameters(), lr=(1e-4+1e-5)/2)
loss_function = nn.CrossEntropyLoss()

if __name__ == "__main__":
    vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)
    accuracy_list = [0]*vertical_shift_range.shape[0]
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
        torch.save(model.state_dict(), 'baseline_aug.pt') 
    else:
        index = 0
        model.load_state_dict(torch.load('baseline_aug.pt', map_location=device))
        model.eval()
        for vsr in vertical_shift_range:
            temp = 0
            total_i = 0
            #if (vsr <= 0.11 and vsr >= 0.09):
                    #print('here')

            for i, (data, y) in enumerate(test_loader):
                #if (data.size()[0] == args.batch_size): #resolve last batch issue later.
                data, y = shift_image(x=data,y=y,width_shift_val=0.0,height_shift_val=vsr)
                y_pred = pred(data)
                #print(y,y_pred)
                #print(y_pred)
                y_temp = y.detach().cpu().numpy()
                aa = accuracy(y_temp,y_pred)
                temp = temp + aa
                total_i = total_i + 1
                    #print(aa)
            print(temp/total_i)
            accuracy_list[index] = temp/total_i
            index = index + 1 
            print(temp/total_i)
        #print(accuracy)
        np.save('BaselineTestVer_aug.npy', accuracy_list)
        plt.plot(vertical_shift_range,accuracy_list)
        plt.show()
        


    
    
