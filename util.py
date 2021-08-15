import torch
from torchvision import datasets, transforms
import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def shift_image(x,y,width_shift_val,height_shift_val):
    #x is assumed to be a tensor of shape (#batch_size, #channels, width, height) = (#batch size, 1, 28, 28)
    #y is assumed to be a tensor of shape (#batch_size)
    batch_size = x.size()[0]
    shift_aa = transforms.RandomAffine(degrees=0,translate=(width_shift_val,height_shift_val))
    x_return = shift_aa(x)
    return x_return,y

def log_gaussian_torch(x,mean,var):
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

def pred(x,model,device):
    #x is assumes to be already in device
    batch_size = x.size()[0]
    yc = np.zeros((10,batch_size))
    for i in range(0,10):
        y = i*torch.ones(batch_size).type(torch.int64).to(device)
        #print(y)
        x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(x,y, manipulated=False)
        #m~q(m|x) for each batch
        std_q2 = torch.exp(0.5*logvar_q2)
        eps_q2 = torch.randn_like(std_q2)
        m = mu_q2 + eps_q2*std_q2 #m size -> (#batch, sizeof(m)) = (#batch, 32)
        #m = torch.zeros(x.size()[0], 32)
        #calculate the log-ration term for each y = c(as an approximation to log p(x,y=c))
        sum = 0
        K = 100
        for j in range(0,K):
            #sum = sum + pred_logRatio(x.to(device), x_recon.to(device), mu_q1.to(device), logvar_q1.to(device), m.to(device))
            sum = sum + pred_logRatio(x, x_recon, mu_q1, logvar_q1, m, device)
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

def pred_logRatio(x, x_recon, mu_q1, logvar_q1, m, device):
    batch_size = x.size()[0]
    #sample from z~q(z|x,y,m)
    std_q1 = torch.exp(0.5*logvar_q1)
    eps_q1 = torch.randn_like(std_q1)
    z = (mu_q1 + eps_q1*std_q1) #size of z -> (#batch_size, sizeof(z))

    #calculate log q(z|x,y,m)
    temp = log_gaussian_torch(z,mu_q1,torch.exp(logvar_q1))
    log_q1 = torch.sum(temp, dim =1)

    #calculate log p(x|y,z,m)
    #log_pxyzm = torch.sum(torch.mul(x.view(-1,784), torch.log(x_recon.view(-1,784)+1e-4)) + torch.mul(1-x.view(-1,784), torch.log(1-x.view(-1,784)+1e-4)), dim=1)
    log_pxyzm = torch.sum(torch.mul(x.view(-1,784), torch.log(x_recon.view(-1,784)+1e-4)) + torch.mul(1-x.view(-1,784), torch.log(1-x_recon.view(-1,784)+1e-4)), dim=1)
    #temp_print = F.binary_cross_entropy(x_recon.view(-1,784),x.view(-1,784),reduction = 'sum')


    #calculate p(y)
    py = 0.1
    
    #calculate log p(z)
    zero_tensor = torch.zeros(z.size()).to(device)
    one_tensor = torch.ones(z.size()).to(device)
    temp = log_gaussian_torch(z,zero_tensor,one_tensor)
    log_pz = torch.sum(temp, dim = 1)

    #adding a constant at the end to prevent underflow. The term will not affect the overall calculation due to the softmax.
    underflow_const = 100
    s = log_pxyzm.reshape(batch_size) + math.log(py) + log_pz.reshape(batch_size) - log_q1.reshape(batch_size) + underflow_const
   
    return torch.exp(s)

def ELBO_x(x,model,device):
    #Calculates ELBO(x)
    yc = torch.ones(x.size()[0]).to(device).type(torch.int64)

    sum = torch.zeros(x.size()[0]).to(device)
    for i in range(0,10):
        yc = i*yc
        #ELBO(x,yc)
        sum = sum + torch.exp(ELBO_xy(x,yc,model))

    return torch.log(sum) 

def ELBO_xy(x, y, model):
    #Calculates ELBO(x,y)
    #x and y should already be in device. 
    x_recon, mu_q1, logvar_q1, mu_q2, logvar_q2 = model(x,y,manipulated=True)

    #p(y)
    py = 0.1

    #calculate  E_q(z,m|x,y)[(log p(x|y,z,m))]. We do monte carlo estimation with just one sample since the batch is large enough.
    BCE = torch.sum(torch.mul(x.view(-1,784), torch.log(x_recon.view(-1,784)+1e-4)) + torch.mul(1-x.view(-1,784), torch.log(1-x_recon.view(-1,784)+1e-4)), dim=1)

    #calculate -1/N sum_N KL(q(z,m|x,y))
    logvar_cat = torch.cat((logvar_q1, logvar_q2), dim = 1)
    mu_cat = torch.cat((mu_q1, mu_q2), dim = 1)
    KLD =  0.5 * torch.sum(1 + logvar_cat - mu_cat.pow(2) - logvar_cat.exp(), dim=1)

    return math.log(py) + BCE + KLD


