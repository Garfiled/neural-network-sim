# coding=utf-8

from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import math

seed = 100
np.random.seed(seed)

X = []
y = []

# 拟合 y = x 直线
# for i in range(-20,20,1):
# 	X.append([i])
# 	y.append([i])

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
for i in range(-50,50,1):
	X.append([i])
	y.append([i**5])

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)


print X.shape
print y.shape

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X_std * 2 - 1
y_std = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
y = y_std * 2 -1


model = Sequential()
model.add(Dense(3, input_dim=1, activation='sigmoid'))
# model.add(Dropout(0.3))
#model.add(LeakyReLU(alpha=.001))
model.add(Dense(1))

model.summary()

# Compile model
adm = Adam(lr=0.03)
model.compile(loss='mean_squared_error', optimizer=adm)


def show_fig(epoch, logs):
	if epoch % 5 == 0:
		# print type(logs),logs
		epoch_his.append(epoch)
		loss_his.append(logs['loss'])
		yHat = model.predict(X)

		ax.clear()
		bx.clear()

		ax.plot(X.reshape(len(X)), y.reshape(len(y)), 'yo')
		ax.plot(X.reshape(len(X)), yHat.reshape(len(y)))

		bx.set_xlim([0, EPOCHS])
		bx.plot(epoch_his, loss_his)

		bx.set_xlabel("epoch:%d loss:%.4f" % (epoch,logs['loss']))
		bx.set_ylabel("loss")

		plt.draw()
		plt.pause(0.01)


EPOCHS = 1000
fig = plt.figure()
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)

epoch_his = []
loss_his = []

model.fit(X,y,epochs=EPOCHS,verbose=1,callbacks=[LambdaCallback(on_epoch_end=show_fig)])

plt.show()