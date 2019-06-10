import os

import h5py
import numpy as np
from scipy.io import wavfile

from synthesis import analyze,synthesis

def convert(f0, sf0, tf0):
    T = len(f0)
    cvf0 = np.zeros(T)
    nonzero_indices = f0 > 0
    cvf0[nonzero_indices] = np.exp((tf0[1] / sf0[1]) *
                                   (np.log(f0[nonzero_indices]) -
                                    sf0[0]) + tf0[0])
    return cvf0


def convert_waveform(file,exten):


    orgstats_h5 = h5py.File("data/stats/SF1.h5", mode='r')
    sf0 = orgstats_h5['f0stats'].value


    tarstats_h5 = h5py.File("data/stats/TF1.h5", mode='r')
    tf0 = tarstats_h5['f0stats'].value
    print(tf0)



    f="100001"
    ###########################################Location of the file
    wavf = "data/wav/SF1/100001.wav"

    print("wave")
    print(wavf)
    fs, x = wavfile.read(wavf)
    x = x.astype(np.float)



    # analyze F0, mcep, and ap
    f0, spc, ap = analyze(x)
    f1 = h5py.File('data/h5/SF1/100001.h5', 'r')
    data_source_test = f1.get('mcep')
    data_source_test = np.array(data_source_test)
    data_source_test = data_source_test.astype('double')

    print(np.shape(data_source_test))
    mcep = data_source_test

    # convert F0
    cvf0 = convert(f0, sf0, tf0)


    data_source_test = file
    data_source_test = np.array(data_source_test)
    data_source_test = data_source_test.astype('double')
    print(np.shape(data_source_test))

    cmcep=data_source_test[0:704]

    wav = synthesis(cvf0,
                                cmcep,
                                ap,
                                r=mcep,
                                alpha=0.41,
                                )
    wavpath = os.path.join("data/test", f + exten)

    wav = np.clip(wav, -32768, 32767)
    wavfile.write(wavpath, fs, wav.astype(np.int16))
    print(wavpath)


