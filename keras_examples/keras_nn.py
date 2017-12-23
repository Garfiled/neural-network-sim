# coding=utf-8

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import LambdaCallback
import keras

seed = 100
np.random.seed(seed)

X = []
y = []

# 拟合 y = x 直线
for i in range(-20, 20, 1):
    X.append([i])
    y.append([i])

# 拟合 y = x**2
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**2])

# 拟合 y = x**3
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**3])


# 拟合 y = sin(x)
# v = -math.pi
# while v <= math.pi:
# 	X.append([v])
# 	y.append([math.sin(v)])
# 	v += 0.2

# 拟合 y = x**4
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**4])

# 拟合 y = x**5
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**5])

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

print X.shape
print y.shape

dest = [-1, 1]

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X_std * (dest[1]-dest[0]) + dest[0]
y_std = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
y = y_std * (dest[1]-dest[0]) + dest[0]

model = Sequential()
# relu 函数总是无端挂掉是怎么回事？
# 数据落在了未激活区域？
# 难道现实使用中 relu函数需要以量取胜
active_func = keras.layers.LeakyReLU(alpha=0.3)
model.add(Dense(1, input_dim=1, activation=active_func
                # kernel_initializer=keras.initializers.zeros(),
                # bias_initializer=keras.initializers.Ones()
                ))
# model.add(Dropout(0.3))
# model.add(LeakyReLU(alpha=.001))
model.add(Dense(1))

model.summary()

# Compile model
opt = optimizers.Adam(lr=0.03)
model.compile(loss='mae', optimizer=opt)


def show_fig(epoch, logs):
    if (epoch % 5 == 0 or epoch == EPOCHS-1):
        epoch_his.append(epoch)
        loss_his.append(logs['loss'])
        yHat = model.predict(X)

        ax.clear()
        bx.clear()

        ax.plot(X.reshape(len(X)), y.reshape(len(y)), 'yo')
        ax.plot(X.reshape(len(X)), yHat.reshape(len(y)))

        bx.set_xlim([0, EPOCHS])
        bx.plot(epoch_his, loss_his)

        bx.set_xlabel("epoch:%d loss:%.4f" % (epoch+1, logs['loss']))
        bx.set_ylabel("loss")

        plt.draw()
        plt.pause(0.01)


EPOCHS = 200
fig = plt.figure()
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)

epoch_his = []
loss_his = []

model.fit(X, y, epochs=EPOCHS,verbose=1, callbacks=[LambdaCallback(on_epoch_end=show_fig)])

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print weights

plt.show()
