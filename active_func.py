import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.arange(-1,1.1,0.1)

# print x
plt.plot(x,sigmoid(x1))
plt.plot(x,x)
plt.show()