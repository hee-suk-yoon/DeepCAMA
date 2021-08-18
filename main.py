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


#  ----------------------------------------------------
parser = argparse.ArgumentParser(description='DeepCAMA MNIST Example')
parser.add_argument('--train', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--ca', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--train-save', type=str2bool, default=False, metavar='T/F')
parser.add_argument('--finetune', type=str2bool, default=False, metavar='T/F')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
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
test_data = datasets.MNIST('./data/train/', train=False, transform=transforms.ToTensor())



#train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)


#---------------------------prepare clean + aug data loader--------------------------------
train_data_clean = list(train_data)
for idx, _ in enumerate(train_data_clean):
    L1 = list(train_data_clean[idx])
    L1.append(0)
    train_data_clean[idx] = tuple(L1)
    
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
    train_data_aug = train_data_aug + split_part
train_data_clean_aug = train_data_clean + train_data_aug
train_loader_clean_aug = torch.utils.data.DataLoader(train_data_clean_aug, batch_size=args.batch_size, shuffle=True, **kwargs)
train_loader_clean = torch.utils.data.DataLoader(train_data_clean, batch_size=args.batch_size, shuffle=True, **kwargs)
#--------------------------------------------------------------------

train_loader_FT = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, **kwargs)
test_loader_FT = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, **kwargs)
# ------------------------------------------------------\


class DeepCAMA(nn.Module):
    def __init__(self):
        super(DeepCAMA, self).__init__()
        #network for q(m|x) (will this part activated during FineTune)
        self.qmx_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_flatten = nn.Flatten()
        self.qmx_fc1 = nn.Linear(1024,500)
        self.qmx_fc21 = nn.Linear(500,32)
        self.qmx_fc22 = nn.Linear(500,32)
    
        #network for q(z|x,y,m)
        self.qzxym_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
        self.qzxym_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
        self.qzxym_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
        self.qzxym_flatten = nn.Flatten()
        self.qzxym_fc1 = nn.Linear(1024,500)
        self.qzxym_fc2 = nn.Linear(500+10+32, 500)
        self.qzxym_fc31 = nn.Linear(500,64)
        self.qzxym_fc32 = nn.Linear(500,64)

        #network for p(x|y,z,m)
        #NNpY
        self.p_fc1 = nn.Linear(10,500)
        self.p_fc2 = nn.Linear(500,500)
        #NNpZ
        self.p_fc3 = nn.Linear(64,500)
        self.p_fc4 = nn.Linear(500,500)
        #NNpM (will leave this part activated during FineTune)
        self.p_fc5 = nn.Linear(32,500)
        self.p_fc6 = nn.Linear(500,500)
        self.p_fc7 = nn.Linear(500,500)
        self.p_fc8 = nn.Linear(500,500)

        self.p_projection = nn.Linear(1500,4*4*64,bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,output_padding=0,stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,output_padding=1,stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=3,padding=1,output_padding=1,stride=2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode1(self,x,y,m):
        #q(z|x,y,m)
        a = F.max_pool2d(F.relu(self.qzxym_conv1(x)),2)
        b = F.max_pool2d(F.relu(self.qzxym_conv2(a)),2)
        c = F.max_pool2d(F.relu(self.qzxym_conv3(b)),2, padding = 1)
        d = self.qzxym_flatten(c)
        d2 = F.relu(self.qzxym_fc1(d))
        e = torch.cat((d2,y,m),dim = 1)
        f = F.relu(self.qzxym_fc2(e))
        return self.qzxym_fc31(f), self.qzxym_fc32(f)

    def encode2(self, x):
        #q(m|x)
        a = F.max_pool2d(F.relu(self.qmx_conv1(x)),2)
        b = F.max_pool2d(F.relu(self.qmx_conv2(a)),2)
        c = F.max_pool2d(F.relu(self.qmx_conv3(b)),2,padding = 1)
        d = self.qmx_flatten(c)
        d2 = F.relu(self.qmx_fc1(d))
        return self.qmx_fc21(d2), self.qmx_fc22(d2)
    
    def decode(self,y,z,m):
        a = F.relu(self.p_fc2(F.relu(self.p_fc1(y))))
        b = F.relu(self.p_fc4(F.relu(self.p_fc3(z))))
        c = F.relu(self.p_fc8(F.relu(self.p_fc7(F.relu(self.p_fc6(F.relu(self.p_fc5(m))))))))

        i = torch.cat((a,b,c), dim = 1)
        i = F.relu(self.p_projection(i))
        j = i.reshape(-1,64,4,4)
        k = F.relu(self.deconv1(j))
        k2 = F.relu(self.deconv2(k))
        return torch.sigmoid(self.deconv3(k2)) 

    def onehot(self,y):
        y_onehot = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
        y_onehot = F.one_hot(y_onehot,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
        y_onehot = y_onehot.to(torch.float32)
        return y_onehot 
    def forward(self, x,y, manipulated):
        #x shape -> (#batch, channels=1, width, height)
        #y shape -> (#batch)
        #print(y)
        
        y_onehot = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
        y_onehot = F.one_hot(y_onehot,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
        y_onehot = y_onehot.to(torch.float32)
        #print(y.size())
        #q(m|x)
        #print(x.size())
        mu_q2, logvar_q2 = self.encode2(x)
        #print(mu_q2)
        m = self.reparameterize(mu_q2, logvar_q2)
        #print(m)
        if not manipulated:
            m = m.zero_()
        #print(m)
        #(q(z|x,y,m))
        mu_q1, logvar_q1 = self.encode1(x,y_onehot,m)
        #print(mu_q1)
        z = self.reparameterize(mu_q1, logvar_q1)

        #p(x|y,z,m)
        x_recon = self.decode(y_onehot,z,m)
        return x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, y, clean):
    #BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar_q1 - mu_q1.pow(2) - logvar_q1.exp())
    batch_size = x.size()[0]
    loss = (1/batch_size)*torch.sum(ELBO_xym0(x,y, model))
    
    alpha = 0.5
    with torch.no_grad():
        ELBO_xym0_calc = ELBO_xym0(x,y,model) #size [128
        ELBO_xy_calc = ELBO_xy(x,y,model,device) #size [128, 128] ###error
    #print(1-clean)
    #print(clean.size())
    loss = torch.sum((1-clean)*ELBO_xym0_calc + clean*ELBO_xy_calc)
    loss.requires_grad = True
    return -loss
    #return -loss


def train(epoch):
    model.eval()
    train_loss = 0
    for batch_id,  (data,y, clean) in enumerate(train_loader_clean):
        data = data.to(device)
        y = y.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        loss = loss_function(data,y,clean)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_id % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader_clean.dataset),
                100. * batch_id / len(train_loader_clean),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader_clean.dataset)))
    return

qmx = [
        'qmx_conv1.weight',
        'qmx_conv1.bias',
        'qmx_conv2.weight',
        'qmx_conv2.bias',
        'qmx_conv3.weight',
        'qmx_conv3.bias',
        'qmx_fc1.weight',
        'qmx_fc1.bias',
        'qmx_fc21.weight',
        'qmx_fc21.bias',
        'qmx_fc22.weight',
        'qmx_fc22.bias'
    ]

NNpM = [
        'p_fc5.weight'
        'p_fc5.bias'
        'p_fc6.weight'
        'p_fc6.bias'
        'p_fc7.weight'
        'p_fc7.bias'
        'p_fc8.weight'
        'p_fc8.bias'
    ]
#Equation (8). The loss function used for fine tuning
def loss_function_FT(x_train, y_train, x_test):
    train_batch_size = x_train.size()[0]
    test_batch_size = x_test.size()[0]
    #alpha = train_batch_size/(train_batch_size+test_batch_size)
    alpha = 0.5
    #print(y_train)
    #ELBO_xym0_calc = ELBO_xym0(x_train,y_train,model)
    with torch.no_grad():
        ELBO_xy_calc = ELBO_xy(x_train,y_train, model,device)
        ELBO_x_calc = ELBO_x(x_test,model,device)

    loss = (alpha*torch.sum(ELBO_xy_calc) + (1-alpha)*torch.sum(ELBO_x_calc))
    loss.requires_grad = True
    return -loss

def finetune(epoch,ready):
    if not(ready):
        for name, param in model.named_parameters():
            #if name == ''
            if (name in qmx) or (name in NNpM):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (data_FT, y_dummy) in enumerate(test_loader_FT):
            data_FT, y_dummy = shift_image(x=data_FT,y=y_dummy,width_shift_val=0.0,height_shift_val=0.3)
            data_FT = data_FT.to(device)
            y_dummy = y_dummy.to(device)
            break
        ready = True

    model.train()
    FT_loss = 0 
    for batch_id,  (data_train,y_train) in enumerate(train_loader_FT):
        data_train = data_train.to(device)
        y_train = y_train.to(device)
        optimizer_FT.zero_grad()
        loss = loss_function_FT(data_train,y_train,data_FT)
        #torch.enable_grad()
        #torch.set_grad_enabled(True)
        loss.backward()
        FT_loss += loss.item()
        optimizer_FT.step()
        if batch_id  % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data_train), len(train_loader_FT.dataset),
                100. * batch_id / len(train_loader_FT),
                loss.item() / len(data_train)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, FT_loss / len(train_loader_FT.dataset)))
    return

def test():
    vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)
    if args.finetune:
        ready = False
        for epoch in range(1,101):
            finetune(epoch,ready)
    torch.save(model.state_dict(), '/media/hsy/DeepCAMA/weight817.pt') 


    model.eval()
    accuracy_list = [0]*vertical_shift_range.shape[0]
    index = 0
    with torch.no_grad():
        for vsr in vertical_shift_range:
            temp = 0
            total_i = 0
            for i, (data, y) in enumerate(test_loader):
                #if (data.size()[0] == args.batch_size): #resolve last batch issue later.
                    #data, y = shift_image(x=data,y=y,width_shift_val=0.0,height_shift_val=vsr)
                data = data.to(device)
                #y = y.to(device)
                data, y = shift_image(x=data,y=y,width_shift_val=0.0,height_shift_val=vsr)
                y_pred = pred(data,model,device)
                y_true = y.detach().cpu().numpy()
                temp = temp + accuracy(y_true,y_pred)
                total_i = total_i + 1
                    #print(aa)
            print(temp/total_i)
            accuracy_list[index] = temp/total_i
            index = index + 1 
            print(temp/total_i)
    return accuracy_list

model = DeepCAMA().to(device)
optimizer = optim.Adam(model.parameters(), lr=(1e-4+1e-5)/2)
optimizer_FT = optim.Adam(model.parameters(), lr = 1e-5)
if __name__ == "__main__":
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)

    #torch.autograd.set_detect_anomaly(True)
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
        if args.train_save:
            torch.save(model.state_dict(), '/media/hsy/DeepCAMA/weight3_3.pt') 
    
    else:
        model.load_state_dict(torch.load('/media/hsy/DeepCAMA/weight3_3.pt', map_location=device))
        model.eval()
        accuracy = test()
    
    print(accuracy)
    #np.save('OurWoFineClean_weight3_2.npy', accuracy_list)
    #plt.plot(vertical_shift_range,accuracy_list)
    #plt.show()
    
    
    
    