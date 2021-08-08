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

parser = argparse.ArgumentParser(description='DeepCAMA MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

"""
p_yi = np.zeros(10)
for i, (data, y) in enumerate(train_data):
    p_yi[y] = p_yi[y] + 1
sum_temp = p_yi.sum()
for i in range(0,10):
    p_yi[i] = p_yi[i]/(sum_temp)
"""

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
        print('size of q(z|x,y,m) a is ' + str(a.size()))
        b = F.max_pool2d(F.relu(self.qzxym_conv2(a)),2)
        print('size of q(z|x,y,m) b is ' + str(b.size()))
        c = F.max_pool2d(F.relu(self.qzxym_conv3(b)),2, padding = 1)
        print('size of q(z|x,y,m) c is ' + str(c.size()))
        #d = torch.flatten(c)
        d = self.qzxym_flatten(c)
        print('size of q(z|x,y,m) d is ' + str(d.size()))
        d2 = F.relu(self.qzxym_fc1(d))
        print('size of q(z|x,y,m) d2 is ' + str(d2.size()))
        print('size of q(z|x,y,m) y is ' + str(y.size()))
        print('size of q(z|x,y,m) m is ' + str(m.size()))
        e = torch.cat((d2,y,m),dim = 1)
        print('size of e is ' + str(e.size()))
        f = F.relu(self.qzxym_fc2(e))
        return self.qzxym_fc31(f), self.qzxym_fc32(f)

    def encode2(self, x):
        #q(m|x)
        a = F.max_pool2d(F.relu(self.qmx_conv1(x)),2)
        print('size of a is ' + str(a.size()))
        b = F.max_pool2d(F.relu(self.qmx_conv2(a)),2)
        c = F.max_pool2d(F.relu(self.qmx_conv3(b)),2,padding = 1)
        #d = torch.flatten(c)
        #d = nn.Flatten(c)
        d = self.qmx_flatten(c)
        print('size of d is ' + str(d.size()))
        d2 = F.relu(self.qmx_fc1(d))
        return self.qmx_fc21(d2), self.qmx_fc22(d2)
    
    def decode(self,y,z,m):
        print('size of decode y is ' + str(y.size()))
        print('size of decode z is ' + str(z.size()))
        print('size of decode m is ' + str(m.size()))
        a = F.relu(self.p_fc2(F.relu(self.p_fc1(y))))
        b = F.relu(self.p_fc4(F.relu(self.p_fc3(z))))
        c = F.relu(self.p_fc8(F.relu(self.p_fc7(F.relu(self.p_fc6(F.relu(self.p_fc5(m))))))))

        i = torch.cat((a,b,c), dim = 1)
        print('size of decode i is ' + str(i.size()))
        i = F.relu(self.p_projection(i))
        print('size of decode i is ' + str(i.size()))
        j = i.reshape(args.batch_size,64,4,4)
        k = F.relu(self.deconv1(j))
        k2 = F.relu(self.deconv2(k))
        return torch.sigmoid(self.deconv3(k2)) 

    def forward(self, x,y):
        #x shape -> (#batch, channels=1, width, height)
        #y shape -> (#batch)
        y = y.view(args.batch_size,-1) #y shape -> (#batch, 1(which is the label))
        y = F.one_hot(y,num_classes=10).view(args.batch_size,10) #y shape -> (#batch, 10)
        y = y.to(torch.float32)
        print('size of y is ' + str(y.size()))
        #q(m|x)
        mu_q2, logvar_q2 = self.encode2(x)
        print('size of mu_q2 is ' + str(mu_q2.size()))
        m = self.reparameterize(mu_q2, logvar_q2)
        print('size of m is ' + str(m.size()))
        #(q(z|x,y,m))
        mu_q1, logvar_q1 = self.encode1(x,y,m)
        z = self.reparameterize(mu_q1, logvar_q1)

        #p(x|y,z,m)
        x_recon = self.decode(y,z,m)
        return x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2


model = DeepCAMA().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#x = torch.ones((1,1,28,28)).to(device)
#a = next(iter(test_loader)) 
#x = a[0][0].reshape(1,1,28,28).to(device)
#save_image(x[0],'tempp.png')
#(batch, in_channel, width, height)
#y = torch.ones(2).to(device)
#y = torch.tensor([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).to(device)
#(label)
#m = torch.ones(32).to(device)
#z = torch.ones(64).to(device)
#x_recon = model(x,y)
#print(x_recon.size())
#print(x_recon)
#save_image(x_recon[0],'temp.png')

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, y, mu_q1, logvar_q1, mu_q2, logvar_q2):
    BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_q1 - mu_q1.pow(2) - logvar_q1.exp())

    return BCE - 0.1 + KLD

def train(epoch):
    model.eval()
    for batch_id,  (data,y) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(data)
        loss = loss_function(x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2)
    return

if __name__ == "__main__":
    """
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    """
    data, y = next(iter(train_loader))
    #y = y.to(torch.float32)
    #print(data.dtype)
    #print(y)
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(data.to(device),y.to(device))
    print(x_recon.size())
    print(mu_q1.size())
    print(logvar_q1.size())