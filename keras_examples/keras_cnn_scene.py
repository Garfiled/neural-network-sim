# coding=utf-8

import keras
from keras.models import Sequential
from keras.layers import Conv2D
import numpy as np

X_train = np.zeros((1,10,10,1))
X_train[0][0][0]=200

y_train = np.zeros((1,10,10,1))
y_train[0][0][0]=200


X_train /= 400
y_train /= 400

print np.matrix(X_train)
print np.matrix(y_train)

model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 3),
                 activation='sigmoid',
                 padding='same',
                 input_shape=(10,10,1)))

# opt = keras.optimizers.Adadelta()
opt = keras.optimizers.RMSprop(0.008)
model.compile(loss='mse',optimizer=opt)

model.summary()

model.fit(X_train, y_train,
          batch_size=1,
          epochs=2000,
          verbose=0)

y_pre = model.predict(X_train)

y_pre *= 400
y_pre = np.asarray(y_pre,dtype=int)
print np.matrix(y_pre)