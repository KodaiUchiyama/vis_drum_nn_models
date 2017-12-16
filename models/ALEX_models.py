# coding:utf-8
import os
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils.io_utils import HDF5Matrix
import h5py
from keras import backend as K
#from exp_models.my_callback import *
import time
#モデルの構築
from keras.applications.inception_v3 import InceptionV3


def AlexNet_model(image_dim, audio_vector_dim, learning_rate, weight_init, output_dim, optimizer):#(224,224,3)timespace-imageセット3枚
    (img_rows, img_cols, img_channels) = image_dim
    input_img = Input(shape=(img_rows, img_cols, img_channels))

    DROPOUT = 0.2

    # Like Hanoi's work with DeepMind Reinforcement Learning, build a model that does not use pooling layers
    # to retain sensitivty to locations of objects
    #物体の位置に関する繊細さを得るためにこのモデルはプーリング層を使わないで行った。
    #CNN layers that increase in filter number, decrease in filter size
    # and decrease in filter stride. The authors reasoned
    # that such a design pattern enables the CNN to be
    # sensitive to the location of small details
    # Tried (64,128,256,512)

    # Channel 1 - Conv Net Layer 1
    x = Conv2D(48, 11, strides=4, activation='relu', padding='same')(input_img)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 1 - Conv Net Layer 2
    x = Conv2D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 1 - Conv Net Layer 3
    x = Conv2D(192, 3, activation='relu', padding='same')(x)
    # Channel 1 - Conv Net Layer 4
    x = Conv2D(192, 3, activation='relu', padding='same')(x)
    # Channel 1 - Cov Net Layer 5
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 1 - Cov Net Layer 6
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)
    x = Reshape((1, 2048))(x)
    x = LSTM(256, input_shape=(1, 2048), dropout=0.2, return_sequences=True)(x)
    x = LSTM(256, dropout=0.2, name='LSTM_reg_output')(x)
    network_output = Dense(output_dim)(x)#最後にオーディオデータの次元数にあわせる

    model = Model(inputs=input_img, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd  = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    #print("learning rate:",learning_rate)

    model.compile(loss=custom_loss, optimizer=optimizer)
    #model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())

    return model
def InceptionV3_model(image_dim, audio_vector_dim, learning_rate, weight_init, output_dim, optimizer):
    #(img_rows, img_cols, img_channels) = image_dim
    #すべての画像はこのサイズに変形される
    img_width, img_height = 299 , 299
    input_shape = (img_height, img_width, 3)

    DROPOUT = 0.2
    base_model = InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    # Channel 1 - Cov Net Layer 8
    x = Dense(1024, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    x = Reshape((1, 1024))(x)
    x = LSTM(256, input_shape=(1, 1024), dropout=0.2, return_sequences=True)(x)
    x = LSTM(256, dropout=0.2, name='LSTM_reg_output')(x)
    network_output = Dense(output_dim)(x)#最後にオーディオデータの次元数にあわせる

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=network_output)
    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())
    return model
#カスタムコスト関数
def custom_loss(y_true, y_pred):
    return K.log(K.sum(K.square(K.abs(y_pred - y_true))) + 1 / (25*25))
