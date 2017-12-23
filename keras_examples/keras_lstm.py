import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,20,20,100],
				 [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 100],
				 [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 100],
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12]]


# Preprocess Data: (This does not work)
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:,:-1] # all but last
target = wholeSequence[:,-1] # all but first


print data.shape
print data
print target.shape
print target

data /= 100
target /= 100

data = data.reshape((data.shape[0], data.shape[1], 1))
print data
# target = target.reshape((-1, 11))

# Build Model
model = Sequential()
model.add(LSTM(1, input_shape=(data.shape[1], data.shape[2])))
model.add(Dense(1))

adam = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mean_absolute_error', optimizer=adam)

model.summary()

model.fit(data, target, epochs=500, batch_size=5, verbose=2)

# Do the output values match the target values?
predict = model.predict(data)
print repr(data)
print repr(predict)