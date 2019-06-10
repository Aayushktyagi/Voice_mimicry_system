import h5py
import numpy as np
import pysptk
from scipy.io import wavfile
from synthesis import analyze, spc2npow

def main():
    wavf = "data\\test\\100001_2.wav"
    fs, x = wavfile.read(wavf)
    x = x.astype(np.float)
    print("Extracting features: ",wavf)
    f0, spc, ap = analyze(x)
    print(np.shape(spc))
    mcep = pysptk.sp2mc(spc, 24, 0.410)
    npow = spc2npow(spc)
    print(np.shape(mcep))
    f1="data\\100001_2.h5"
    h5 = h5py.File(f1, mode='w')
    print("hhbjbj")
    print(np.shape(f0))
    print(np.shape(spc))
    print(np.shape(ap))
    print(np.shape(mcep))
    h5.create_dataset('f0', data=f0)
    h5.flush()
    h5.create_dataset('mcep', data=mcep)
    h5.flush()
    h5.create_dataset('npow', data=npow)
    h5.flush()

if __name__ == '__main__':
    main()
