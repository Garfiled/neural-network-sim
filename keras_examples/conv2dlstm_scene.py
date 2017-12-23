# coding=utf-8

import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import keras


def create_model():
    model = Sequential()

    model.add(ConvLSTM2D(filters=2, kernel_size=(1, 1),
                         input_shape=(None, 10, 10, 1),
                         padding='same', return_sequences=False))

    # model.add(BatchNormalization())
    #
    model.add(Conv2D(filters=1, input_shape=(10,10,1),kernel_size=(1, 1),
                     activation='linear',
                     # kernel_initializer=keras.initializers.zeros(),
                     # bias_initializer=keras.initializers.zeros(),
                     padding='same', data_format='channels_last'))

    # opt = optimizers.RMSprop(lr=0.005)
    opt = optimizers.adam(lr=0.03)
    # opt = optimizers.sgd(lr=0.8)
    model.compile(loss='mae', optimizer=opt)

    model.summary()

    return model


if __name__ == "__main__":
    X_train = np.zeros((7,7,10,10,1))
    X_train[0][0][0][0][0] = 200
    X_train[0][1][0][0][0] = 200
    X_train[0][2][0][0][0] = 200
    X_train[0][3][0][0][0] = 200
    X_train[0][4][0][0][0] = 100
    X_train[0][5][0][0][0] = 20
    X_train[0][6][0][0][0] = 20

    X_train[1][0][0][0][0] = 200
    X_train[1][1][0][0][0] = 200
    X_train[1][2][0][0][0] = 200
    X_train[1][3][0][0][0] = 100
    X_train[1][4][0][0][0] = 20
    X_train[1][5][0][0][0] = 20
    X_train[1][6][0][0][0] = 200

    X_train[2][0][0][0][0] = 200
    X_train[2][1][0][0][0] = 200
    X_train[2][2][0][0][0] = 100
    X_train[2][3][0][0][0] = 20
    X_train[2][4][0][0][0] = 20
    X_train[2][5][0][0][0] = 200
    X_train[2][6][0][0][0] = 200

    X_train[3][0][0][0][0] = 200
    X_train[3][1][0][0][0] = 100
    X_train[3][2][0][0][0] = 20
    X_train[3][3][0][0][0] = 20
    X_train[3][4][0][0][0] = 200
    X_train[3][5][0][0][0] = 200
    X_train[3][6][0][0][0] = 200

    X_train[4][0][0][0][0] = 100
    X_train[4][1][0][0][0] = 20
    X_train[4][2][0][0][0] = 20
    X_train[4][3][0][0][0] = 200
    X_train[4][4][0][0][0] = 200
    X_train[4][5][0][0][0] = 200
    X_train[4][6][0][0][0] = 200

    X_train[5][0][0][0][0] = 20
    X_train[5][1][0][0][0] = 20
    X_train[5][2][0][0][0] = 200
    X_train[5][3][0][0][0] = 200
    X_train[5][4][0][0][0] = 200
    X_train[5][5][0][0][0] = 200
    X_train[5][6][0][0][0] = 100

    X_train[6][0][0][0][0] = 20
    X_train[6][1][0][0][0] = 200
    X_train[6][2][0][0][0] = 200
    X_train[6][3][0][0][0] = 200
    X_train[6][4][0][0][0] = 200
    X_train[6][5][0][0][0] = 100
    X_train[6][6][0][0][0] = 20

    y_train = np.zeros((7,10,10,1))
    y_train[0][0][0][0] = 200
    y_train[1][0][0][0] = 200
    y_train[2][0][0][0] = 200
    y_train[3][0][0][0] = 200
    y_train[4][0][0][0] = 100
    y_train[5][0][0][0] = 20
    y_train[6][0][0][0] = 20

    X_train = np.asarray(X_train,dtype=float)
    y_train = np.asarray(y_train,dtype=float)

    X_train /= 200
    y_train /= 200

    print X_train.shape
    print y_train.shape

    model = create_model()

    model.fit(X_train,y_train,epochs=500,batch_size=1)

    y_pre = model.predict(X_train)

    y_pre *= 200

    y_train *= 200

    for i in range(y_train.shape[0]):
        print np.matrix(y_train[i])
        print np.matrix(np.asarray(y_pre[i], dtype=int))



