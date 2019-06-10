import h5py
import os
import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, GRU,LSTM
from keras.layers.wrappers import TimeDistributed,Bidirectional
from reshape import reshape
#data loading 
f=h5py.File('data/alligned_data/it2_jnt1_x25.h5','r')
data_source=f.get('jnt')
data_source=np.array(data_source)
print(data_source.shape)
f1=h5py.File('data/alligned_data/it2_jnt2_x25.h5','r')
data_target=f1.get('jnt')
data_target=np.array(data_target)
print(data_target.shape)

# #keras implimentation
batch_size=1
tsteps = 50
data_dim = 25
#epochs
epochs = 50
# #data spliting 
data_source_train=data_source[:100000,:]
data_source_valid=data_source[100000:,:]
data_target_train=data_target[:100000,:]
data_target_valid=data_target[100000:,:]
# #zeros padding
# #source data 
data_source_train= reshape(data_source_train, tsteps, data_dim)
data_source_valid = reshape(data_source_valid, tsteps, data_dim)
#target data
data_target_train= reshape(data_target_train, tsteps, data_dim)
data_target_valid = reshape(data_target_valid, tsteps, data_dim)
#checking shape
print('source data',np.shape(data_source_train))
print('target data',np.shape(data_target_train))
#model selection
model = Sequential()
model.add(LSTM(units=70,
              batch_input_shape=(batch_size, tsteps, data_dim),
              return_sequences=True,
              stateful=True))
model.add(LSTM(70, return_sequences=True, stateful=True))
# model.add(LSTM(70, stateful=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(data_dim)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)
print('Training')
epoch = list(range(epochs))
loss = []
val_loss = []

for i in range(epochs):
  print('Epoch', i, '/', epochs)
  history = model.fit(data_source_train,
                      data_target_train,
                      batch_size=batch_size,
                      verbose=1,
                      epochs=1,
                      shuffle=False,
                      validation_data=(data_source_valid, data_target_valid))

  loss.append(history.history['loss'])
  val_loss.append(history.history['val_loss'])

model.reset_states()

#saving model
model.save_weights('model/LSTM.h5')

with open('model/LSTM.json', 'w') as model_json:
  model_json.write(model.to_json())



