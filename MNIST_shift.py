def shift_image(x,y,width_shift_val,height_shift_val):
    #x is assumed to be a tensor of shape (#batch_size, #channels, width, height) = (#batch size, 1, 28, 28)
    #y is assumed to be a tensor of shape (#batch_size, 1)
    x = x.detach().cpu().numpy().reshape(x.size()[0],28,28)
    y = y.detach().cpu().numpy().reshape(y.size()[0])

    # import relevant library
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # create the class object
    datagen = ImageDataGenerator(width_shift_range=width_shift_val, height_shift_range=height_shift_val)
    # fit the generator
    datagen.fit(x.reshape(x.shape[0], 28, 28, 1))

    a = datagen.flow(x.reshape(x.shape[0], 28, 28, 1),y.reshape(y.shape[0], 1),batch_size=x.shape[0],shuffle=False)

    X, Y = next(iter(a))     
    return X,Y