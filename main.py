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
#  ----------------------------------------------------


parser = argparse.ArgumentParser(description='DeepCAMA MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
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


model = DeepCAMA().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, y, mu_q1, logvar_q1, mu_q2, logvar_q2):
    BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_q1 - mu_q1.pow(2) - logvar_q1.exp())

    return BCE  + KLD

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

def pred_logRatio(x, x_recon, mu_q1, logvar_q1, m):
    batch_size = x.size()[0]
    #sample from z~q(z|x,y,m)
    std_q1 = torch.exp(0.5*logvar_q1)
    eps_q1 = torch.randn_like(std_q1)
    z = (mu_q1 + eps_q1*std_q1) #size of z -> (#batch_size, sizeof(z))

    #calculate log q(z|x,y,m)
    #log_q1 = torch.zeros((batch_size,1))
    temp = log_gausian_torch(z,mu_q1,torch.exp(logvar_q1))
    log_q1 = torch.sum(temp, dim =1)
    #log_q1 = log_q1.cpu()
    """
    for i in range(0,batch_size):
        sum_temp = 0
        for j in range(0,mu_q1.size()[1]):
            temp = log_gaussian(z[i][j].item(),mu_q1[i][j].item(),math.exp(logvar_q1[i][j].item()))
            sum_temp = sum_temp + temp
        log_q1[i][0] = sum_temp
    """
    #calculate log p(x|y,z,m)
    log_pxyzm = torch.zeros((x.size()[0],1)).to(device)
    for i in range(0,batch_size):
        log_pxyzm[i][0] = torch.sum(torch.mul(x[i].view(-1,784), torch.log(x_recon[i].view(-1,784)+1e-4)) + torch.mul(1-x[i].view(-1,784), torch.log(1-x[i].view(-1,784)+1e-4)))

    #calculate p(y)
    py = 0.1
    
    #calculate log p(z)
    zero_tensor = torch.zeros(z.size()).to(device)
    one_tensor = torch.ones(z.size()).to(device)
    temp = log_gausian_torch(z,zero_tensor,one_tensor)
    log_pz = torch.sum(temp, dim = 1)

    """
    log_pz = torch.zeros((batch_size,1))
    for i in range(0,batch_size):
        sum_temp = 0
        for j in range(0,z.size()[1]):
            temp = log_gaussian(z[i][j].item(),0,1)
            sum_temp = sum_temp + temp
        log_pz[i][0] = sum_temp
    """

    #adding a constant at the end to prevent underflow. The term will not affect the overall calculation due to the softmax.
    underflow_const = 100
    s = log_pxyzm.reshape(batch_size) + math.log(py) + log_pz.reshape(batch_size) - log_q1.reshape(batch_size) + underflow_const
   
    return torch.exp(s)

def pred(x):
    batch_size = x.size()[0]
    yc = np.zeros((10,batch_size))
    for i in range(0,10):
        y = i*torch.ones(128).type(torch.int64)
        #print(y)
        x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(x.to(device),y.to(device), manipulated=False)
        #m~q(m|x) for each batch
        std_q2 = torch.exp(0.5*logvar_q2)
        eps_q2 = torch.randn_like(std_q2)
        m = mu_q2 + eps_q2*std_q2 #m size -> (#batch, sizeof(m)) = (#batch, 32)
        #m = torch.zeros(x.size()[0], 32)
        #calculate the log-ration term for each y = c(as an approximation to log p(x,y=c))
        sum = 0
        K = 20
        for j in range(0,K):
            sum = sum + pred_logRatio(x.to(device), x_recon.to(device), mu_q1.to(device), logvar_q1.to(device), m.to(device))
            #print(j)
        #print(sum)
        log_pxy = torch.log(sum).view(x.size()[0]).detach().cpu().numpy()
        yc[i] = log_pxy
    
    #print(yc)
    exp_yc = np.exp(yc)
    #print(exp_yc)
    sum_exp_yc = np.sum(exp_yc,axis=0)

    for i in range(0,batch_size):
        for j in range(0,10):
            exp_yc[j][i] = exp_yc[j][i]/sum_exp_yc[i]

    label = np.argmax(exp_yc,axis=0)
    return label
    



if __name__ == "__main__":
    
    #for epoch in range(1, args.epochs + 1):
    #    train(epoch)
    
    #torch.save(model.state_dict(), '/media/hsy/DeepCAMA/weight.pt')
    
    
    model.load_state_dict(torch.load('/media/hsy/DeepCAMA/weight.pt', map_location=device))
    model.eval()

    a,y = next(iter(test_loader)) 
    print(y)
    #y[0] = 7
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(a.to(device),y.to(device), manipulated=False)
    #print(a)

    y_pred = pred(a)
    print(y_pred)

    y_temp = y.detach().cpu().numpy()
    print(accuracy(y_temp,y_pred))
    #print(x_recon[1])
    #save_image(a[8].view(1,28,28),'actual.png')
    #save_image(x_recon[8].view(1,28,28),'temp1.png')
    



    """
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(a.to(device),y.to(device), manipulated=False)
    save_image(x_recon[0].view(1,28,28),'temp2.png')
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(a.to(device),y.to(device), manipulated=False)
    save_image(x_recon[0].view(1,28,28),'temp3.png')
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(a.to(device),y.to(device), manipulated=False)
    save_image(x_recon[0].view(1,28,28),'temp4.png')
    """
