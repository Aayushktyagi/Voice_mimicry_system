import h5py
import numpy as np

from keras.models import model_from_json

from reshape import reshape
from convert_nn import convert_waveform
batch_size=1
tsteps = 50
data_dim = 25


print('loadig model')
with open('model/BidirLSTM.json', 'r') as model_json:
  model = model_from_json(model_json.read())

model.load_weights('model/BidirLSTM.h5')


f=h5py.File('data/h5/100001.h5','r')
data_source_test=f.get('jnt')
data_source_test=np.array(data_source_test)
print(data_source_test.shape)

data_source_test = reshape(data_source_test, tsteps, data_dim)
print('Predicting')
prediction_test = model.predict(data_source_test, batch_size=batch_size)
print(prediction_test.shape)
prediction_test = prediction_test.reshape(-1, data_dim)
print(prediction_test.shape)


convert_waveform(prediction_test,"'bidir_lstm.wav'")