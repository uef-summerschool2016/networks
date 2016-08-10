from __future__ import print_function
import scipy.io as sio
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, merge, Convolution1D, MaxPooling1D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

np.random.seed(1337)

# Define model
input = Input(shape=(1000,4))
conv = Convolution1D(nb_filter=64, filter_length=26, activation='relu')(input)
pool = MaxPooling1D(pool_length=13, stride=13)(conv)
drop1 = Dropout(0.2)(pool)
forward_lstm = LSTM(output_dim=128, return_sequences=True, consume_less='gpu')(drop1)
backward_lstm = LSTM(output_dim=128, return_sequences=True, go_backwards=True, consume_less='gpu')(drop1)
merged = merge([forward_lstm, backward_lstm], mode='concat', concat_axis=-1)
drop2 = Dropout(0.5)(merged)
flat = Flatten()(drop2)
dense = Dense(output_dim=1024, activation='relu')(flat)
output = Dense(output_dim=4, activation='sigmoid')(dense)
model = Model(input=input, output=output)

# Optimizer
optim = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8)

# Compile
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

# Callback
checkpointer = ModelCheckpoint(filepath='seq_example_best_weights.hdf5', verbose=1, save_best_only=True)

# Load data
d = sio.loadmat('seq_data.mat')
data = d['data']
labels = d['labels']

history = model.fit(data, labels, batch_size=32, nb_epoch=10, verbose=1, shuffle=True, validation_split=0.25, callbacks=[checkpointer])

