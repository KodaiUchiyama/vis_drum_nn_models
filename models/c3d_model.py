# coding:utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam
from keras.layers import LSTM
from keras.utils.io_utils import HDF5Matrix
import h5py
from keras import backend as K
import os
'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]Theano
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]tensorflow
'''

def get_model(summary=False, backend='tf'):
    """ Return the Keras model of the network
    """
    model = Sequential()
    if backend == 'tf':
        input_shape=(30, 90, 160, 3) # l, h, w, c
    else:
        input_shape=(3, 30, 90, 160) # c, l, h, w
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    return model
def add_LSTM(model, output_dim):
    #model.add(Dense(5, activation='softmax', name='fc9'))
    #model.add(LSTM(256, input_shape=(1, 4096), dropout=0.2, return_sequences=True))
    #model.add(LSTM(256, dropout=0.2, name='LSTM_reg_output'))
    model.add(Reshape(output_dim, input_shape=(4096,)))
    model.add(Dense(output_dim))

    print(model.summary())
    return model
def get_int_model(model, backend='tf'):

    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w

    int_model = Sequential()

    int_model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape,
                            weights=model.layers[0].get_weights(),
                            trainable=False))
    int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))

    # 2nd layer group
    int_model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2',
                            weights=model.layers[2].get_weights(),
                            trainable=False))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))

    # 3rd layer group
    int_model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a',
                            weights=model.layers[4].get_weights(),
                            trainable=False))
    int_model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b',
                            weights=model.layers[5].get_weights(),
                            trainable=False))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))

    # 4th layer group
    int_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a',
                            weights=model.layers[7].get_weights(),
                            trainable=False))
    int_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4b',
                            weights=model.layers[8].get_weights(),
                            trainable=False))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))

    # 5th layer group
    int_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5a',
                            weights=model.layers[10].get_weights(),
                            trainable=False))
    int_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5b',
                            weights=model.layers[11].get_weights(),
                            trainable=False))
    int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad'))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))

    int_model.add(Flatten())
    # FC layers group
    int_model.add(Dense(4096, activation='relu', name='fc6',
                            weights=model.layers[15].get_weights()))
    int_model.add(Dropout(.5))
    int_model.add(Dense(4096, activation='relu', name='fc7',
                            weights=model.layers[17].get_weights()))
    int_model.add(Dropout(.5))

    return int_model

if __name__ == '__main__':
    model = get_model(summary=True)
