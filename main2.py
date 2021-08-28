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
from torch.autograd import Variable

#  ----------------------------------------------------
parser = argparse.ArgumentParser(description='DeepCAMA MNIST Example')

parser.add_argument('--train-clean', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--run', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--train', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--train-save', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--loss-plot-save', type=str2bool, required=True, metavar='T/F')
parser.add_argument('--finetune', type=str2bool, default=False, metavar='T/F')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
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
#train_data, val_data = torch.utils.data.random_split(train_data, [int(0.95*len(train_data)),int(0.05*len(train_data))], generator=torch.Generator().manual_seed(42))
test_data = datasets.MNIST('./data/train/', train=False, transform=transforms.ToTensor())



#train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)
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


train_data_clean, val_data_clean = torch.utils.data.random_split(train_data_clean, [int(0.95*len(train_data_clean)),int(0.05*len(train_data_clean))], generator=torch.Generator().manual_seed(42))
train_data_clean_aug, val_data_clean_aug = torch.utils.data.random_split(train_data_clean_aug, [int(0.95*len(train_data_clean_aug)),int(0.05*len(train_data_clean_aug))], generator=torch.Generator().manual_seed(42))
train_loader_clean_aug = torch.utils.data.DataLoader(train_data_clean_aug, batch_size=args.batch_size, shuffle=True, **kwargs)
train_loader_clean = torch.utils.data.DataLoader(train_data_clean, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader_clean = torch.utils.data.DataLoader(val_data_clean, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader_clean_aug = torch.utils.data.DataLoader(val_data_clean_aug, batch_size=args.batch_size, shuffle=True, **kwargs)
#--------------------------------------------------------------------


#---------------------------prepare data for finetune--------------------------------
test_data_finetune = list(test_data)
#for idx, _ in enumerate(test_data_finetune):
#    L1 = list(test_data_finetune[idx])
#    L1.append(1)
#    test_data_finetune[idx] = tuple(L1)
listofones_1 = [1] * 2
split_list_1 = [int(element * 0.5*len(test_data_finetune)) for element in listofones_1]
test_data_split = torch.utils.data.random_split(test_data_finetune, split_list_1, generator=torch.Generator().manual_seed(42))
test_data_half = list(test_data_split[0])
vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)


ft_data_aug = []
for idx in range(0,10):
    ft_data_aug_split = []
    for idx2, _ in enumerate(test_data_half):
        shift_fn = transforms.RandomAffine(degrees=0,translate=(0.0,vertical_shift_range[idx]))
        L1 = list(test_data_half[idx2])
        L1[0] = shift_fn(L1[0])
        L1.append(1)
        ft_data_aug_split.append(tuple(L1))
    ft_data_aug = ft_data_aug + ft_data_aug_split
ft_data = ft_data_aug + list(train_data_clean)
#ft_data, ft_val_aug = torch.utils.data.random_split(ft_data, [int(0.95*len(ft_data)),int(0.05*len(ft_data))], generator=torch.Generator().manual_seed(42))
ft_data, ft_val_aug = torch.utils.data.random_split(ft_data, [int(0.90*len(ft_data)),int(0.10*len(ft_data))], generator=torch.Generator().manual_seed(42))
finetune_data_loader = torch.utils.data.DataLoader(ft_data, batch_size=args.batch_size, shuffle=True, **kwargs)
finetune_val_loader = torch.utils.data.DataLoader(ft_val_aug,  batch_size=args.batch_size, shuffle=True, **kwargs)

#--------------------------------------------------------------------





#train_loader_FT = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, **kwargs)
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
        #y_onehot = y_onehot.to(torch.float64)
        return y_onehot 
    def forward(self, x,y=0, manipulated=False, z_sampled = 0, m_sampled = 0, phase = 0):
        """
            phase:
                0 -> run through the whole network
                1 -> run through q1 (q(z|x,y,m)) (encoder) 
                2 -> run through q2 (z(m|x) (encoder)
                3 -> run through p (decoder)
        """
        if phase == 0:
            y_onehot = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
            y_onehot = F.one_hot(y_onehot,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
            y_onehot = y_onehot.to(torch.float32)
            #y_onehot = y_onehot.to(torch.float64)

            mu_q2, logvar_q2 = self.encode2(x)

            m = self.reparameterize(mu_q2, logvar_q2)

            if not manipulated:
                m = m.zero_()

            mu_q1, logvar_q1 = self.encode1(x,y_onehot,m)

            z = self.reparameterize(mu_q1, logvar_q1)

            x_recon = self.decode(y_onehot,z,m)
            return x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2
        elif phase == 1:
            y_onehot = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
            y_onehot = F.one_hot(y_onehot,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
            y_onehot = y_onehot.to(torch.float32)
            #y_onehot = y_onehot.to(torch.DoubleTensor)
            #y_onehot = y_onehot.to(torch.float64)

            mu_q1, logvar_q1 = self.encode1(x,y_onehot,m_sampled)
            
            z = self.reparameterize(mu_q1, logvar_q1)
            return mu_q1, logvar_q1, z 
        
        elif phase == 2:
            mu_q2, logvar_q2 = self.encode2(x)
            
            m = self.reparameterize(mu_q2, logvar_q2)
            return mu_q2, logvar_q2, m 
        
        elif phase == 3:
            y_onehot = y.view(-1,1) #y shape -> (#batch, 1(which is the label))
            y_onehot = F.one_hot(y_onehot,num_classes=10).view(-1,10) #y shape -> (#batch, 10)
            y_onehot = y_onehot.to(torch.float32)
            #y_onehot = y_onehot.to(torch.float64)

            x_recon = self.decode(y_onehot,z_sampled,m_sampled)
            return x_recon
            
            
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, y, clean):
    
    alpha = 0.5

    num_aug_data = torch.sum(clean)
    num_clean_data = clean.size()[0] - num_aug_data
    mu_q2, logvar_q2, m_sampled = model(x=x,phase=2)
    
    if num_aug_data == 0:
        loss = torch.sum(ELBO_xym0(x,y,model)) 
    elif num_clean_data == 0:
        #loss = torch.sum((1/num_aug_data)*ELBO_xy_calc)
        #loss = torch.sum((1/num_aug_data)* ELBO_xy(x,y,model,device))
        loss = torch.sum(ELBO_xy(x,y,model,device))
    else:
        loss = torch.sum((1-clean)*(1/num_clean_data)*ELBO_xym0(x,y,model) + clean*(1/num_aug_data)*ELBO_xy(x,y,model,device))

    #loss.requires_grad = True
    #print(loss.grad_fn)
    return -loss
    
    #return -loss


def train(epoch):
    train_loss = 0
    if args.train_clean:
        dataloader = train_loader_clean
    else: 
        dataloader = train_loader_clean_aug
        
    for batch_id,  (data,y, clean) in enumerate(dataloader):
        data = data.to(device)
        y = y.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        for num_sampling in range(0,1):
            loss = loss_function(data,y,clean)
            train_loss += loss.item()
            loss.backward()
        train_loss = train_loss/(num_sampling+1)
        optimizer.step()
        if batch_id % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(dataloader.dataset),
                100. * batch_id / len(dataloader),
                loss.item() / len(data)))

    average_loss = train_loss / len(dataloader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, average_loss))
    return average_loss
def val_ft(epoch):
    validation_loss = 0
    with torch.no_grad():
        for batch_id, (data,y,clean) in enumerate(finetune_val_loader):
            ELBOx = True
            if (data.size()[0] == args.batch_size):
                data = data.to(device)
                y = y.to(device)
                clean = clean.to(device)
                loss = loss_function_FT(data,y,clean,ELBOx)
                validation_loss += loss.item()
                ELBOx = False
                loss = loss_function_FT(data,y,clean,ELBOx)
                validation_loss += loss.item()
    
    average_loss = validation_loss / len(finetune_val_loader.dataset)
    print('====> Epoch: {} Average validation loss: {:.4f}'.format(
          epoch, average_loss))
    return average_loss
def val(epoch):
    if args.train_clean:
        dataloader = val_loader_clean
    else: 
        dataloader = val_loader_clean_aug
    validation_loss = 0
    with torch.no_grad():
        for batch_id, (data,y,clean) in enumerate(dataloader):
            data = data.to(device)
            y = y.to(device)
            clean = clean.to(device)
            loss = loss_function(data,y,clean)
            validation_loss += loss.item()
    
    average_loss = validation_loss / len(dataloader.dataset)
    print('====> Epoch: {} Average validation loss: {:.4f}'.format(
          epoch, average_loss))
    return average_loss

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
def loss_function_FT(x,y,test,ELBOx):
    #with torch.no_grad():
    alpha = 0.5

    num_test_data = torch.sum(test)
    num_clean_data = (test.size()[0] - num_test_data)
    if num_test_data == 0:
        loss = (1/num_clean_data)*torch.sum(ELBO_xy(x,y,model,device))
    elif num_clean_data == 0:
        loss = (1/num_test_data)*torch.sum(ELBO_x(x,model,device))
    else:
        #print('here')
        #print((1-test)*(1/num_clean_data)*ELBO_xy(x,y,model,device))
        #print(test*(1/num_test_data)*ELBO_x(x,model,device))
        if ELBOx:
            loss = torch.sum(test*(1/num_test_data)*ELBO_x(x,model,device))
            #print('1' + str(loss))
            #ELBOx = False
        else:
            loss = torch.sum((1-test)*(1/num_clean_data)*ELBO_xy(x,y,model,device))
            #print('2' + str(loss))
            #print('here2')
            #ELBOx = True
        #loss = torch.sum((1-test)*(1/num_clean_data)*ELBO_xy(x,y,model,device) + test*(1/num_test_data)*ELBO_x(x,model,device))

    #loss = alpha*(1/train_batch_size)*torch.sum(ELBO_xy(x_train,y_train, model,device) + (1-alpha)*(1/test_batch_size)*torch.sum(ELBO_x(x_test,model,device)))
    #loss = loss.to(device)
    #print('here')
    #print(loss)
    #print(model)
    return -loss

def finetune(epoch):
    model.train()
    FT_loss = 0 
    
    for batch_id, (data,y,train) in enumerate(finetune_data_loader):
    #for batch_id, (data,y,train) in enumerate(finetune_data_loader):
        ELBOx = True
        if (data.size()[0] == args.batch_size):
            loss_temp = 0
            data = data.to(device)
            y = y.to(device)
            train = train.to(device)
            optimizer_FT.zero_grad()
            loss = loss_function_FT(data,y,train,ELBOx)
            loss.backward()
            FT_loss += loss.item()
            loss_temp += loss.item()
            ELBOx = False
            #loss = loss_function_FT(data,y,train,ELBOx)
            #loss.backward()
            """
            for param in model.parameters():
                if param.requires_grad == True:
                    print("param.data",torch.isfinite(param.data).all())
                    print("param.grad.data",torch.isfinite(param.grad.data).all(),"\n")
                    print(param.grad.data)
            """
            #print('backward stop')
            FT_loss += loss.item()
            loss_temp += loss.item()
            optimizer_FT.step()


            if batch_id  % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(finetune_data_loader.dataset),
                    100. * batch_id / len(finetune_data_loader),
                    loss_temp / len(data)))
    average_loss = FT_loss / len(finetune_data_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch,average_loss))
    return average_loss

def test():
    #-------------------------------------finetune-------------------------------------
    vertical_shift_range = np.arange(start=0.0,stop=1.0,step=0.1)
    if args.finetune:
        for name, param in model.named_parameters():
            #if name == ''
            if (name in qmx) or (name in NNpM):
                param.requires_grad = True
            else:
                param.requires_grad = False
        ft_loss =[]
        val_loss = []
        for epoch in range(1,2001):
            ft_loss.append(finetune(epoch))
            s = '/media/hsy/DeepCAMA/ft_weights_2/TrainCleanEpoches' + str(epoch) + '.pt'
            val_loss.append(val_ft(epoch))
            #torch.save(model.state_dict(), s) 
            ft_loss = np.array(ft_loss)
            np.save('ft_loss_2_temp.npy',ft_loss)
            ft_loss = list(ft_loss)
            
            val_loss = np.array(val_loss)
            np.save('val_loss_2_temp.npy',val_loss)
            val_loss = list(val_loss)

    #---------------------------------------test--------------------------------------
    model.eval()
    accuracy_list = [0]*vertical_shift_range.shape[0]
    index = 0
    with torch.no_grad():
        for vsr in range(0,10):
            temp = 0
            total_i = 0
            s = '/media/hsy/DeepCAMA/data_loader/test_shift' + str(vsr) +'.pt'
            test_data_shifted = torch.load(s)
            test_loader = torch.utils.data.DataLoader(test_data_shifted,  batch_size=args.batch_size, shuffle=True, **kwargs)
            for i, (data, y, dummy) in enumerate(test_loader):
                print(i)
                data = data.to(device)
                y_pred = pred(data,model,device)
                y_true = y.detach().cpu().numpy()
                temp = temp + accuracy(y_true,y_pred)
                total_i = total_i + 1
            print(temp/total_i)
            accuracy_list[index] = temp/total_i
            index = index + 1 
            print(temp/total_i)
    return accuracy_list





























#----------------------------------------------main-----------------------------------------------------#
model = DeepCAMA().to(device)
optimizer = optim.Adam(model.parameters(), lr=(1e-4+1e-5)/2)
optimizer_FT = optim.Adam(model.parameters(), lr = 1e-4)
if __name__ == "__main__":
    if args.run: 
        if args.train:
            #model.load_state_dict(torch.load('/media/hsy/DeepCAMA/weight821.pt', map_location=device))
            loss_train_values = []
            loss_val_values = []
            for epoch in range(1, args.epochs + 1):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                        loss_train_values.append(train(epoch))
                        if args.train_save:
                            s = '/media/hsy/DeepCAMA/main2_weights/TrainCleanEpoches' + str(epoch) + '.pt'
                            torch.save(model.state_dict(), s)                        
                    elif phase == 'val': 
                        model.eval()
                        loss_val_values.append(val(epoch))
                    #if epoch >=1 and epoch <= 1600:

                if args.loss_plot_save:
                    loss_train_values = np.array(loss_train_values)
                    loss_val_values = np.array(loss_val_values)
                    np.save('/media/hsy/DeepCAMA/results/main2_loss_train_values.npy',loss_train_values)
                    np.save('/media/hsy/DeepCAMA/results/main2_loss_val_values.npy',loss_val_values)
                    loss_train_values = list(loss_train_values)
                    loss_val_values = list(loss_val_values)
                        
        
        else:
            model.load_state_dict(torch.load('/media/hsy/DeepCAMA/main2_weights/TrainCleanEpoches650.pt', map_location=device))
            #model.load_state_dict(torch.load('/media/hsy/DeepCAMA/ft_weights_2/TrainCleanEpoches1200.pt', map_location=device))
            #model.load_state_dict(torch.load('/media/hsy/DeepCAMA/baseline.pt'))
            #model.load_state_dict(torch.load('/media/hsy/DeepCAMA/weight3_2.pt', map_location=device))
            model.eval()
            accuracy = test()
            print(accuracy)
            #np.save('/media/hsy/DeepCAMA/results/ft_weights/Epoches300_test.npy', accuracy)
        
    #np.save('OurWoFineClean_weight3_2.npy', accuracy_list)
    #plt.plot(vertical_shift_range,accuracy_list)
    #plt.show()
    
    
    
    