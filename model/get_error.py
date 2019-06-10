import h5py
import numpy as np
from fastdtw import fastdtw
import os
def melcd(a1, a2):

    diff = a1 - a2
    mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))

    return mcd

def estimate_twf(S1, T1):
    _, tw1 = fastdtw(S1, T1, dist=melcd)
    twf = np.array(tw1).T
    return twf

def mcd_db(a1,a2):
    diff = a1 - a2
    mcd = 10.0 / np.log(10) \
          * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    return mcd


def aligndata(S1, T1):
    S = S1
    T = T1
    twf = estimate_twf(S, T)
    mcd = mcd_db(S[twf[0]], T[twf[1]])
    return S[twf[0]],T[twf[1]], twf, mcd




def data_allign(dir1,dir2):
    f1 = h5py.File(dir1, 'r')
    d1 = f1.get('mcep')
    d1 = np.array(d1)
    d1 = d1.astype('double')
    f1 = h5py.File(dir2, 'r')
    d2 = f1.get('mcep')
    d2 = np.array(d2)
    d2 = d2.astype('double')
    data1, data2, _, mcd = aligndata(d1, d2)
    return data1, data2

def mcd_diff(dir1,dir2):
    f1 = h5py.File(dir1, 'r')
    d1 = f1.get('mcep')
    d1 = np.array(d1)
    d1 = d1.astype('double')
    f1 = h5py.File(dir2, 'r')
    d2 = f1.get('mcep')
    d2 = np.array(d2)
    d2 = d2.astype('double')
    data1, data2, _, mcd = aligndata(d1, d2)
    return mcd
err=[]
dir1 = "data/100001.h5"
dir2 = "data/100001'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100002.h5"
dir2 = "data/100002'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100003.h5"
dir2 = "data/100003'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100004.h5"
dir2 = "data/100004'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100005.h5"
dir2 = "data/100005'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100006.h5"
dir2 = "data/100006'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100007.h5"
dir2 = "data/100007'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100008.h5"
dir2 = "data/100008'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100009.h5"
dir2 = "data/100009'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100010.h5"
dir2 = "data/100010'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100011.h5"
dir2 = "data/100011'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100012.h5"
dir2 = "data/100012'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100013.h5"
dir2 = "data/100013'bl.h5"
err.append(mcd_diff(dir1,dir2))

dir1 = "data/100014.h5"
dir2 = "data/100014'bl.h5"
err.append(mcd_diff(dir1,dir2))


dir1 = "data/100015.h5"
dir2 = "data/100015'bl.h5"
err.append(mcd_diff(dir1,dir2))

avgerr = 0
for i in range(len(err)):
    avgerr = avgerr + err[i]

print(avgerr / 15)