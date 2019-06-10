import os

import h5py
from sklearn.externals import joblib
import numpy as np
import pysptk
from scipy.io import wavfile
from GMM import  GMMConvertor
from synthesis import analyze,synthesis
from delta import static_delta
from gv import GV
mcepgv = GV()


def convert(f0, sf0, tf0):
    T = len(f0)
    cvf0 = np.zeros(T)
    nonzero_indices = f0 > 0
    cvf0[nonzero_indices] = np.exp((tf0[1] / sf0[1]) *
                                   (np.log(f0[nonzero_indices]) -
                                    sf0[0]) + tf0[0])
    return cvf0


def main():


    os_h5 = h5py.File("data/stats/SF1.h5", mode='r')
    sf0 = os_h5['f0stats'].value
    cgv_h5 = h5py.File("data/model/cvgv.h5", mode='r')
    cvgvs = cgv_h5['cvgv'].value
    ts_h5 = h5py.File("data/stats/TF1.h5", mode='r')

    tf0 = ts_h5['f0stats'].value
    tgv_h5 = ts_h5['gv'].value
    print(tf0)
    mpath = 'model/GMM.pkl'
    mcepgmm = GMMConvertor(n_mix=32,
                           covtype="full"
                           )
    param = joblib.load(mpath)
    mcepgmm.open_from_param(param)



    f="100001"
    ###########################################Location of the file
    wavf = "data/wav/SF1/100001.wav"

    print("wave")
    print(wavf)
    fs, x = wavfile.read(wavf)
    x = x.astype(np.float)



    # analyze F0, mcep, and ap
    f0, spc, ap = analyze(x)
    mcep = pysptk.sp2mc(spc, 24, 0.410)

    mcep_0th = mcep[:, 0]
    cvmcep_wopow = mcepgmm.convert(static_delta(mcep[:, 1:]),
                                   cvtype="mlpg")
    cvmcep = np.c_[mcep_0th, cvmcep_wopow]


    # convert F0
    cvf0 = convert(f0, sf0, tf0)
    cvmwGV = mcepgv.postfilter(cvmcep,
                                   tgv_h5,
                                   cvgvstats=cvgvs,
                                   startdim=1)




    wav = synthesis(cvf0,
                                cvmwGV,
                                ap,
                                r=mcep,
                                alpha=0.41,
                                )
    wavpath = os.path.join("data/test", f + '_gmm.wav')

    wav = np.clip(wav, -32768, 32767)
    wavfile.write(wavpath, fs, wav.astype(np.int16))
    print(wavpath)

if __name__ == '__main__':
    main()
