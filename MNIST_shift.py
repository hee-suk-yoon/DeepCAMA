def shift_image(x):
    #x is assumed to be a tensor of shape (#batch_size, #channels, width, height) = (#batch size, 1, 28, 28)
    x.detach().cpu().numpy().reshape(x.size()[0],28,28)
    return