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
parser.add_argument('--train-save', type=str2bool, default=False, metavar='T/F')
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


train_loader_FT = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, **kwargs)
test_loader_FT = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, **kwargs)
# ------------------------------------------------------\


class DeepCAMA(nn.Module):
    def __init__(self):
        super(DeepCAMA, self).__init__()

        #network for q(m|x)
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
        self.p_fc1 = nn.Linear(10,500)
        self.p_fc2 = nn.Linear(500,500)
        self.p_fc3 = nn.Linear(64,500)
        self.p_fc4 = nn.Linear(500,500)
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

    def forward(self, x,y, manipulated):
        #x shape -> (#batch, channels=1, width, height)
        #y shape -> (#batch)
        y = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
        y = F.one_hot(y,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
        y = y.to(torch.float32)

        #q(m|x)
        mu_q2, logvar_q2 = self.encode2(x)
        m = self.reparameterize(mu_q2, logvar_q2)
        if not manipulated:
            m = m.zero_()

        #(q(z|x,y,m))
        mu_q1, logvar_q1 = self.encode1(x,y,m)
        z = self.reparameterize(mu_q1, logvar_q1)

        #p(x|y,z,m)
        x_recon = self.decode(y,z,m)
        return x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, y, mu_q1, logvar_q1, mu_q2, logvar_q2):
    BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_q1 - mu_q1.pow(2) - logvar_q1.exp())

    return BCE  + KLD

#Equation (8). The loss function used for fine tuning
def loss_function_FT(x_train, y_train, x_test):
    train_batch_size = x_train.size()[0]
    test_batch_size = x_test.size()[0]

    alpha = train_batch_size/(train_batch_size+test_batch_size)
    ELBO_xy_calc = ELBO_xy(x_train,y_train)
    ELBO_x_calc = ELBO_x(x_test)

    loss = alpha*(1/train_batch_size)*ELBO_xy_calc + (1-alpha)*(1/test_batch_size)*ELBO_x_calc
    return loss

def train(epoch):
    model.eval()
    train_loss = 0
    for batch_id,  (data,y) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(data,y,manipulated=False)
        loss = loss_function(x_recon, data, y, mu_q1, logvar_q1, mu_q2, logvar_q2)
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

def test():
    vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)
    accuracy_list = [0]*vertical_shift_range.shape[0]
    index = 0
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
if __name__ == "__main__":

    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
        if args.train_save:
            torch.save(model.state_dict(), '/media/hsy/DeepCAMA/weight.pt') 
    
    else:
        model.load_state_dict(torch.load('/media/hsy/DeepCAMA/weight3_2.pt', map_location=device))
        model.eval()
        accuracy = test()
 
    print(accuracy)
   #np.save('OurWoFineClean_weight3_2.npy', accuracy_list)
    plt.plot(vertical_shift_range,accuracy_list)
    plt.show()
    
    
    
    