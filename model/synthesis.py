

import numpy as np
import pyworld
import pysptk
def spp(sv):

    f2 = len(sv) - 1
    f1 = f2 * 2
    p = sv[0] + sv[f2]
    for i in range(1, f2):
        p += 2.0 * sv[i]
    p =p/f1

    return p


def analyze(x):
    x = np.array(x, dtype=np.float)
    f0, time_axis = pyworld.harvest(x, 16000, f0_floor=40,
                                    f0_ceil=500, frame_period=5)

    spc = pyworld.cheaptrick(x, f0, time_axis, 16000,
                             fft_size=1024)
    ap = pyworld.d4c(x, f0, time_axis, 16000, fft_size=1024)
    return f0, spc, ap


def synthesis(f0, mcep, ap, r=None, alpha=0.42):
    if r is not None:
        mcep = mod_p(mcep, r)
    spc = pysptk.mc2sp(mcep, alpha, 1024)
    wav = pyworld.synthesize(f0, spc, ap,
                             16000, frame_period=5)

    return wav

def mod_p(m, r):
    m1 = pysptk.mc2e(m, alpha=0.42, irlen=256)
    r1 = pysptk.mc2e(r, alpha=0.42, irlen=256)
    p = np.log(r1 / m1) / 2
    t_m = np.copy(m)
    t_m[:, 0] =t_m[:, 0]+ p
    return t_m

def spc2npow(sp):
    npow = np.apply_along_axis(spp, 1, sp)
    mpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / mpow)
    return npow
