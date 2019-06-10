
import h5py
import numpy as np
from sklearn.externals import joblib
from GMM import GMMConvertor,GMMTrainer
from delta import static_delta
from gv import GV
def fconversion(omceps, gmm, gmmmode=None):
    gmm1 = GMMConvertor(n_mix=32,
                         covtype='full',
                         gmmmode=gmmmode,
                         )
    gmm1.open_from_param(gmm.param)

    cmceps = []
    for mcep in omceps:
        mcep_0th = mcep[:, 0]
        cvmcep = gmm1.convert(static_delta(mcep[:, 1:]),
                               cvtype='full')
        cvmcep = np.c_[mcep_0th, cvmcep]
        cmceps.append(cvmcep)

    return cmceps


def main():
    f = h5py.File('data/alligned_data/it2_jnt1_x25.h5', 'r')
    data_source = f.get('jnt')
    data_source = np.array(data_source)
    jnt=data_source

    gmm = GMMTrainer(n_mix=32, n_iter=100,
                     covtype='full')
    gmm.train(jnt)
    joblib.dump(gmm.param, "model/GMM.pkl")

    f1 = h5py.File('data/h5/SF1/100001.h5', 'r')
    data_source_test = f1.get('mcep')
    d1 = np.array(data_source_test)
    mcep = d1.astype('double')

    cmceps = fconversion(mcep, gmm, gmmmode=None)

    gv = GV()
    cvgvstats = gv.estimate(cmceps)
    f1='data/model/cvgv.h5'
    h5 = h5py.File(f1, mode='w')
    h5.create_dataset('f0', data=cvgvstats)
    h5.flush()



if __name__ == '__main__':
    main()
