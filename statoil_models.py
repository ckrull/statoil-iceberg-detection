from keras import applications, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, GlobalAveragePooling2D,
                          MaxPooling2D)
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, Adam
from keras.preprocessing.image import ImageDataGenerator


def Simple_CNN(input_shape,optimizer=Adam(lr=0.0005),depth=1,width=1,batch_norm=False):
    """returns a simple keras-based CNN (3 conv blocks 32,32,64 filters)

    The model assumes a binary classification task (binary crossentropy loss function). Using a Softmax activation function for the top layer
   
    Parameters
    ----------
    input_shape : tuple
        int tuple of the shape of the data to be used
    optimizer : keras.optimizer, optional
        the optimizer (the default is Adam(lr=0.0005))
    depth : int, optional
        the number of fully connected layers (the default is 1)
    width : int, optional
        the number of nodes of the fully connected layers *64 (the default is 1, which means 64 nodes)
    batch_norm : bool, optional
        add a batch normalization layer after each convolutional and fully connected layer (the default is False, which [default_description])
   
    Returns
    -------
    keras.model
        the compiled model, ready to be trained
    """

    simple_model = Sequential()
    simple_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    simple_model.add(Activation('relu',))
    if batch_norm:
        simple_model.add(BatchNormalization())
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))

    simple_model.add(Conv2D(32, (3, 3)))
    simple_model.add(Activation('relu'))
    if batch_norm:
        simple_model.add(BatchNormalization())
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))

    simple_model.add(Conv2D(64, (3, 3)))
    simple_model.add(Activation('relu'))
    if batch_norm:
        simple_model.add(BatchNormalization())
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))
    simple_model.add(Flatten())
    for _ in range(depth):
        simple_model.add(Dense(64*width))
        simple_model.add(Activation('relu'))
        if batch_norm:
            simple_model.add(BatchNormalization())
    simple_model.add(Dropout(0.5))
    simple_model.add(Dense(1))
    simple_model.add(Activation('sigmoid'))
    simple_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return simple_model 


def Larger_CNN(input_shape,optimizer=Adam(lr=0.0005),depth=1,width=1,batch_norm=False):
    """returns a slightly larger keras-based CNN (4 conv blocks 16,32,64,128)

    The model assumes a binary classification task (binary crossentropy loss function). Using a Softmax activation function for the top layer
   
    Parameters
    ----------
    input_shape : tuple
        int tuple of the shape of the data to be used
    optimizer : keras.optimizer, optional
        the optimizer (the default is Adam(lr=0.0005))
    depth : int, optional
        the number of fully connected layers (the default is 1)
    width : int, optional
        the number of nodes of the fully connected layers *64 (the default is 1, which means 64 nodes)
    batch_norm : bool, optional
        add a batch normalization layer after each convolutional and fully connected layer (the default is False, which [default_description])
   
    Returns
    -------
    keras.model
        the compiled model, ready to be trained
    """
    larger_model = Sequential()
    larger_model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    larger_model.add(Activation('relu',))
    if batch_norm:
        larger_model.add(BatchNormalization())
    larger_model.add(MaxPooling2D(pool_size=(2, 2)))

    larger_model.add(Conv2D(32, (3, 3)))
    larger_model.add(Activation('relu'))
    if batch_norm:
        larger_model.add(BatchNormalization())
    larger_model.add(MaxPooling2D(pool_size=(2, 2)))

    larger_model.add(Conv2D(64, (3, 3)))
    larger_model.add(Activation('relu'))
    if batch_norm:
        larger_model.add(BatchNormalization())
    larger_model.add(MaxPooling2D(pool_size=(2, 2)))
    larger_model.add(Conv2D(128, (3, 3)))
    larger_model.add(Activation('relu'))
    if batch_norm:
        larger_model.add(BatchNormalization())
    larger_model.add(MaxPooling2D(pool_size=(2, 2)))
    larger_model.add(Flatten())
    for _ in range(depth):
        larger_model.add(Dense(64*width))
        larger_model.add(Activation('relu'))
        if batch_norm:
            larger_model.add(BatchNormalization())
    larger_model.add(Dropout(0.5))
    larger_model.add(Dense(1))
    larger_model.add(Activation('sigmoid'))
    larger_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return larger_model 


def vgg16_finetune(input_shape):
    """return a VGG16 model with the last convolutional block and the fully connected layers trainable

    the top fully connected layer was trained on the statoil bottleneck features 
    
    Parameters
    ----------
    input_shape : tuple
        the dimensions of the model input array
    """

    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
    top_model.load_weights('VGG16_bottle.h5')

# add the model on top of the convolutional base
    vgg16_model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
    for layer in vgg16_model.layers[:15]:
        layer.trainable = False
    
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
    vgg16_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=5e-6, momentum=0.9),
              metrics=['accuracy'])
    return vgg16_model
