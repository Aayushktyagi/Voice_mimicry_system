# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import sklearn.mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

from delta import construct_static_and_delta_matrix


class GMMTrainer(object):


    def __init__(self, n_mix=32, n_iter=100, covtype='full'):
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.covtype = covtype

        # construct GMM parameter
        self.param = sklearn.mixture.GaussianMixture(
            n_components=self.n_mix,
            covariance_type=self.covtype,
            max_iter=self.n_iter)

        if self.covtype == 'block_diag':
            raise NotImplementedError()

    def train(self, jnt):
        if self.covtype == 'full':
            self.param.fit(jnt)
        elif self.covtype == 'block_diag':
            self._train_block_diag(jnt)

        return

    def train_singlepath(self, reference_jnt, jnt):

        raise NotImplementedError()
        pass

    def _train_block_diag(self, jnt):

        raise NotImplementedError()
        pass


class GMMConvertor(object):

    def __init__(self, n_mix=32, covtype='full', gmmmode=None):
        self.n_mix = n_mix
        self.covtype = covtype
        self.gmmmode = gmmmode

    def open_from_param(self, param):
        self.param = param
        self._deploy_parameters()

        return

    def convert(self, data, cvtype='mlpg'):
        cseq, wseq, mseq, covseq = self._gmmmap(data)

        if cvtype == 'mlpg':
            # maximum likelihood parameter generation
            odata = self._mlpg(mseq, covseq)
        elif cvtype == 'mmse':
            # minimum mean square error based parameter generation
            odata = self._mmse(wseq, data)
        else:
            raise ValueError('please choose conversion mode in `mlpg`, `mmse`')

        return odata

    def _gmmmap(self, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        # estimate posterior sequence
        wseq = self.pX.predict_proba(sddata)

        # estimate mixture sequence
        cseq = np.argmax(wseq, axis=1)

        mseq = np.zeros((T, sddim))
        covseq = np.zeros((T, sddim, sddim))
        for t in range(T):
            # read maximum likelihood mixture component in frame t
            m = cseq[t]

            # conditional mean vector sequence
            mseq[t] = self.meanY[m] + \
                self.A[m] @ (sddata[t] - self.meanX[m])

            # conditional covariance sequence
            covseq[t] = self.cond_cov_inv[m]

        return cseq, wseq, mseq, covseq

    def _mmse(self, wseq, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        odata = np.zeros((T, sddim))
        for t in range(T):
            for m in range(self.n_mix):
                odata[t] += wseq[t, m] * \
                    (self.meanY[m] +
                     self.A[m] @ (sddata[t] - self.meanX[m]))

        # retern static and throw away delta component
        return odata[:, :sddim // 2]

    def _mlpg(self, mseq, covseq):
        # parameter for sequencial data
        T, sddim = mseq.shape

        # prepare W
        W = construct_static_and_delta_matrix(T, sddim // 2)

        # prepare D
        D = get_diagonal_precision_matrix(T, sddim, covseq)

        # calculate W'D
        WD = W.T @ D

        # W'DW
        WDW = WD @ W

        # W'Um
        WDm = WD @ mseq.flatten()

        # estimate y = (W'DW)^-1 * W'Dm
        odata = scipy.sparse.linalg.spsolve(
            WDW, WDm, use_umfpack=False).reshape(T, sddim // 2)

        # return odata
        return odata

    def _deploy_parameters(self):
        # read JD-GMM parameters from self.param
        self.w = self.param.weights_
        self.jmean = self.param.means_
        self.jcov = self.param.covariances_

        # devide GMM parameters into source and target parameters
        sddim = self.jmean.shape[1] // 2
        self.meanX = self.jmean[:, 0:sddim]
        self.meanY = self.jmean[:, sddim:]
        self.covXX = self.jcov[:, :sddim, :sddim]
        self.covXY = self.jcov[:, :sddim, sddim:]
        self.covYX = self.jcov[:, sddim:, :sddim]
        self.covYY = self.jcov[:, sddim:, sddim:]

        # change model paramter of GMM into that of gmmmode
        if self.gmmmode is None:
            pass
        elif self.gmmmode == 'diff':
            self._transform_gmm_into_diffgmm()
        elif self.gmmmode == 'intra':
            self._transform_gmm_into_intragmm()
        else:
            raise ValueError('please choose GMM mode in [None, diff, intra]')

        # estimate parameters for conversion
        self._set_Ab()
        self._set_pX()

        return

    def _set_Ab(self):
        # calculate A and b from self.jmean, self.jcov
        sddim = self.jmean.shape[1] // 2

        # calculate inverse covariance for covariance XX in each mixture
        self.covXXinv = np.zeros((self.n_mix, sddim, sddim))
        for m in range(self.n_mix):
            self.covXXinv[m] = np.linalg.inv(self.covXX[m])

        # calculate A, b, and conditional covariance given X
        self.A = np.zeros((self.n_mix, sddim, sddim))
        self.b = np.zeros((self.n_mix, sddim))
        self.cond_cov_inv = np.zeros((self.n_mix, sddim, sddim))
        for m in range(self.n_mix):
            # calculate A (i.e., A = yxcov_m * xxcov_m^-1)
            self.A[m] = self.covYX[m] @ self.covXXinv[m]

            # calculate b (i.e., b = mean^Y - A * mean^X)
            self.b[m] = self.meanY[m] - self.A[m] @ self.meanX[m]

            # calculate conditional covariance
            # (i.e., cov^(Y|X)^-1 = (yycov - A * xycov)^-1)
            self.cond_cov_inv[m] = np.linalg.inv(self.covYY[
                m] - self.A[m] @ self.covXY[m])

        return

    def _set_pX(self):
        # probability density function of X
        self.pX = sklearn.mixture.GaussianMixture(
            n_components=self.n_mix, covariance_type=self.covtype)
        self.pX.weights_ = self.w
        self.pX.means_ = self.meanX
        self.pX.covariances_ = self.covXX

        # following function is required to estimate porsterior
        # P(X | \lambda^(X)))
        self.pX.precisions_cholesky_ = _compute_precision_cholesky(
            self.covXX, self.covtype)

        return

    def _transform_gmm_into_diffgmm(self):
        self.meanX = self.meanX
        self.meanY = self.meanY - self.meanX
        self.covXX = self.covXX
        self.covYY = self.covXX + self.covYY - self.covXY - self.covYX
        self.covXY = self.covXY - self.covXX
        self.covYX = self.covXY.transpose(0, 2, 1)
        return

    def _transform_gmm_into_intragmm(self):
        self.meanX = self.meanX
        self.meanY = self.meanX
        self.covXX = self.covXX
        self.covXY @= np.linalg.solve(self.covYY, self.covYX)
        self.covYX = self.covXY
        self.covYY = self.covXX
        return


def get_diagonal_precision_matrix(T, D, covseq):
    return scipy.sparse.block_diag(covseq, format='csr')
