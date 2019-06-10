import numpy as np

def reshape(a, tsteps, data_dim):
  zpad_size = int((np.ceil(a.shape[0] / tsteps) * tsteps) - a.shape[0])
  zeros_matrix = np.zeros((zpad_size, a.shape[1]))
  a = np.concatenate((a, zeros_matrix))
  return a.reshape((int(a.shape[0] / tsteps), tsteps, data_dim))