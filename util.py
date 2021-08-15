import torch
from torchvision import datasets, transforms
import numpy as np
def shift_imaged(x,y,width_shift_val,height_shift_val):
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

def shift_image_v2(x,y,width_shift_val,height_shift_val):
    #x is assumed to be a tensor of shape (#batch_size, #channels, width, height) = (#batch size, 1, 28, 28)
    #y is assumed to be a tensor of shape (#batch_size)
    batch_size = x.size()[0]
    shift_aa = transforms.RandomAffine(degrees=0,translate=(width_shift_val,height_shift_val))
    x_return = shift_aa(x)
    return x_return,y
