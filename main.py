from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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

class DeepCAMA(nn.Module):
    def __init__(self):
        super(DeepCAMA, self).__init__()

        #network for q(m|x)
        self.qmx_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding='same')
        self.qmx_fc1 = nn.Linear(1024,500)
        self.qmx_fc21 = nn.Linear(500,32)
        self.qmx_fc22 = nn.Linear(500,32)
    
        #network for q(z|x,y,m)
        self.qzxym_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
        self.qzxym_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
        self.qzxym_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding='same')
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
        c = F.max_pool2d(F.relu(self.qzxym_conv3(b)),2)
        d = torch.flatten(c)
        d2 = F.relu(self.qzxym_fc1(d))
        e = torch.cat((d2,y,m))
        f = F.relu(self.qzxym_fc2(e))
        return self.qzxym_fc31(f), self.qzxym_fc32(f)

    def encode2(self, x):
        #q(m|x)
        a = F.max_pool2d(F.relu(self.qmx_conv1(x)),2)
        b = F.max_pool2d(F.relu(self.qmx_conv2(a)),2)
        c = F.max_pool2d(F.relu(self.qmx_conv3(b)),2,padding = 1)
        d = torch.flatten(c)
        d2 = F.relu(self.qmx_fc1(d))
        return self.qmx_fc21(d2), self.qmx_fc22(d2)
    
    def decode(self,y,z,m):
        a = F.relu(self.p_fc2(F.relu(self.p_fc1(y))))
        b = F.relu(self.p_fc4(F.relu(self.p_fc3(z))))
        c = F.relu(self.p_fc8(F.relu(self.p_fc7(F.relu(self.p_fc6(F.relu(self.p_fc5(m))))))))

        i = torch.cat((a,b,c))
        i = F.relu(self.p_projection(i))
        j = i.reshape(1,64,4,4)
        k = F.relu(self.deconv1(j))
        k2 = F.relu(self.deconv2(k))
        return torch.sigmoid(self.deconv3(k2)) 

    def forward(self, x,y):

        #q(m|x)
        mu_q2, logvar_q2 = self.encode2(x)
        m = self.reparameterize(mu_q2, logvar_q2)

        #(q(z|x,y,m))
        mu_q1, logvar_q1 = self.encode1(x,y,m)
        z = self.reparameterize(mu_q1, logvar_q1)

        #p(x|y,z,m)
        x_recon = self.decode(y,z,m)
        return x_recon


model = DeepCAMA().to(device)
x = torch.ones((1,1,28,28)).to(device)
a = next(iter(test_loader)) 
x = a[0][0].reshape(1,1,28,28).to(device)
save_image(x[0],'tempp.png')
#(batch, in_channel, width, height)
#y = torch.ones(2).to(device)
y = torch.tensor([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).to(device)
#(label)
#m = torch.ones(32).to(device)
#z = torch.ones(64).to(device)
x_recon = model(x,y)
print(x_recon.size())
print(x_recon)
save_image(x_recon[0],'temp.png')
"""
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            print(data)
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))




if __name__ == "__main__":
    

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #torch.save(model.state_dict(), '/media/hsy/VariationalAutoEncoder/weight.pt')
        
        test(epoch)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
    

    model.load_state_dict(torch.load('/media/hsy/VariationalAutoEncoder/weight.pt', map_location=device))
    model.eval()
    a = next(iter(test_loader)) 
    #print(len(a))
    print(a[0][0].shape)
    plt.imshow(a[0][0].reshape(28,28))
    plt.show()
    save_image(a[0][0].view(1,28,28),'hello.png')

    b= a[0].to(device)
    print(b)
    for i in range(1,10):
        #out = model(b)
        recon_batch, mu, logvar = model(a[0][0])
        print('image' + str(i))
        print(recon_batch.view(1,28,28))
        s = 'recon' + str(i) + '.png'
        save_image(recon_batch.view(1,28,28),s)

"""