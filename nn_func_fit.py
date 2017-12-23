# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from sklearn.utils import shuffle
import math


class Neuron_Layer():
    alpha = 0.001

    def __init__(self, lastLayerSize, layerSize, active_func_name='sigmoid'):
        self.size = layerSize
        self.weight = np.random.randn(lastLayerSize, layerSize) * 5
        self.bias = np.random.randn(layerSize)
        print "hidden layer init weight and bias:"
        print self.weight, self.bias

        if active_func_name == 'sigmoid':
            self.active = self.sigmoid
            self.activePrime = self.sigmoidPrime
        elif active_func_name == 'relu':
            self.active = self.relu
            self.activePrime = self.reluPrime
        elif active_func_name == 'leakyrelu':
            self.active = self.leakyRelu
            self.activePrime = self.leakReluPrime
        elif active_func_name == 'linear':
            self.active = self.linear
            self.activePrime = self.linearPrime
        elif active_func_name == 'tanh':
            self.active = self.tanh
            self.activePrime = self.tanhPrime
        else:
            print "not define ", active_func_name
            exit(-1)

    def cal(self, X):
        self.z = np.dot(X, self.weight) + self.bias
        self.a = self.active(self.z)

        return self.a

    def step(self, z):
        return np.where(z >= 0.0, 1.0, 0.0)

    def stepPrime(self, z):
        return np.where(z != 0, 0, np.Inf)

    def linear(self, z):
        return z

    def linearPrime(self, z):
        return 1

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def tanh(self, z):
        return 2 / (1 + np.exp(-2 * z)) - 1

    def tanhPrime(self, z):
        return 1 - self.tanh(z) ** 2

    def relu(self, z):
        return np.where(z >= 0, z, 0.0)

    def reluPrime(self, z):
        return np.where(z >= 0.0, 1.0, 0.0)

    def leakyRelu(self, z):
        return np.where(z < 0.0, self.alpha * z, z)

    def leakReluPrime(self, z):
        return np.where(z > 0.0, 1.0, self.alpha)


class Neural_Network():
    def __init__(self, conf):
        # 定义超参数
        self.learn_rate = conf['learn_rate']
        self.inputLayerSize = conf['input_size']
        self.batch_size = conf['batch_size']
        self.hiddenLayer = []

        print conf

    def addLayer(self, layerSize, active_func_name):
        if len(self.hiddenLayer) == 0:
            lastLayerSize = self.inputLayerSize
        else:
            lastLayerSize = self.hiddenLayer[-1].size

        layer = Neuron_Layer(lastLayerSize, layerSize, active_func_name)

        self.hiddenLayer.append(layer)

    def forward(self, X):
        tmp = X
        for h in self.hiddenLayer:
            tmp = h.cal(tmp)
        # print "forward z:",h.z
        # print "forward a:", h, tmp

        return tmp

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):

        dJd = []
        for i in range(len(self.hiddenLayer) - 1, -1, -1):
            h = self.hiddenLayer[i]
            if i == len(self.hiddenLayer) - 1:
                delta = np.multiply(-(y - self.yHat), h.activePrime(h.z))
            else:
                delta = np.dot(delta, self.hiddenLayer[i + 1].weight.T) * h.activePrime(h.z)

            if i > 0:
                dJdW = np.dot(self.hiddenLayer[i - 1].a.T, delta)
            else:
                dJdW = np.dot(X.T, delta)

            dJdB = np.array(map(sum, zip(*delta)))
            # print "dJdW:",dJdW
            # print "dJdB:",dJdB
            dJd.append((dJdW, dJdB))

        # print dJd
        return dJd

    def train(self, X, y):
        cost = self.costFunction(X, y)
        dJd = self.costFunctionPrime(X, y)

        for i, h in enumerate(self.hiddenLayer):
            r = len(self.hiddenLayer) - 1 - i

            h.weight -= self.learn_rate * dJd[r][0]
            h.bias -= self.learn_rate * dJd[r][1]

        # print "weight:",i,h.weight
        # print "bias:",i,h.bias

        return cost

    def trainSGD(self, X, y):
        X_train, y_train = shuffle(X, y)

        minibatch_size = self.batch_size
        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            cost = self.costFunction(X_train_mini, y_train_mini)
            dJd = self.costFunctionPrime(X_train_mini, y_train_mini)

            for i, h in enumerate(self.hiddenLayer):
                r = len(self.hiddenLayer) - 1 - i

                h.weight -= self.learn_rate * dJd[r][0]
                h.bias -= self.learn_rate * dJd[r][1]

        return cost

    def predict(self, X):
        yHat = self.forward(X)
        return yHat


X = []
y = []

# 拟合 y = x 直线
# for i in range(-20,20,1):
#     X.append([i])
#     y.append([i])

# 拟合 y = x**2
for i in range(-50,50,1):
	X.append([i])
	y.append([i**2])

# 拟合 y = x**3
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**3])


# 拟合 y = sin(x)
# v = -math.pi
# while v <= math.pi:
#     X.append([v])
#     y.append([math.sin(v)])
#     v += 0.2

# 拟合 y = x**4
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**4])

# 拟合 y = x**5
# for i in range(-50,50,1):
# 	X.append([i])
# 	y.append([i**5])

# x = 0
# while x <= 1:
# 	X.append([x])
# 	y.append([0.2 + 0.4*x**2 + 0.3*x*math.sin(15*x) + 0.05*math.cos(50*x)])
# 	x += 0.01

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

print X.shape
print y.shape

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X_std * 2 - 1
y_std = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
y = y_std * 2 - 1

print "X:", X
print "y:", y

conf = dict(
    learn_rate=0.001,
    input_size=1,
    batch_size=1
)

nn = Neural_Network(conf)

nn.addLayer(3, 'sigmoid')
nn.addLayer(1, 'linear')

fig = plt.figure()
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)
epoch_his = []
loss_his = []


def update(epoch):
    loss = nn.trainSGD(X, y)
    # loss = nn.train(X, y)

    epoch_his.append(epoch)
    loss_his.append(loss)

    print "epoch:", epoch, loss

    ax.clear()
    bx.clear()
    # ax.axis([0, 1,0,1])
    ax.plot(X.reshape(len(X)), y.reshape(len(y)), 'yo')

    W1 = nn.hiddenLayer[0].weight
    B1 = nn.hiddenLayer[0].bias

    xs = np.arange(-1, 1.2, 0.2)

    # 第一层神经元输出结果
    for i in range(nn.hiddenLayer[0].size):
        # ys = W1[0][i] * xs + B1[i]
        ys = nn.hiddenLayer[0].active(W1[0][i] * xs + B1[i])
    # ax.plot(xs, ys)

    yHat = nn.forward(X)
    # print "shape:"
    # print X.shape
    # print yHat.shape
    pre = ax.plot(X.reshape(len(X)), yHat.reshape(len(yHat)))
    plt.setp(pre, color='r', linewidth=2.0)

    bx.set_xlim([0, EPOCHS])
    bx.plot(epoch_his, loss_his)

    bx.set_xlabel("epoch:%d loss:%.4f" % (epoch, loss))
    bx.set_ylabel("loss")


EPOCHS = 200

aimation = anim.FuncAnimation(fig, update, interval=300, frames=range(EPOCHS), repeat=False)

plt.xlabel('X')
plt.ylabel('y')

plt.show()
