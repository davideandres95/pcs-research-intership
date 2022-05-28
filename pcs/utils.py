import numpy as np
import scipy.stats
import torch
from scipy.optimize import fminbound
from pcs import helper as hlp
import matplotlib.pyplot as plt


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        for key in self.__dict__.keys():
            s = s + key + ':' + str(self.__dict__[key]) + '\n'

        return s


def hotOnes(size, tanspose, M, seed=None):
    if seed != None:
        np.random.seed(seed)
    x_seed = np.eye(M, dtype=int)
    idx = np.random.randint(M, size=size)
    x = np.transpose(x_seed[:, idx], tanspose)
    return x, idx, x_seed


def lin2dB(lin, dBtype='dBm'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10 * np.log10(lin) - fact


def dB2lin(dB, dBtype='dBm'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10 ** ((dB + fact) / 10)


def p_norm(p, x, fun=lambda x: torch.pow(torch.abs(x), 2)):
    return torch.sum(p * fun(x))


def get_py_on_x(chi, y, N0):
    return np.power(2 * np.pi * N0, -0.5) * np.exp(-np.square(y + chi) / (2 * N0))

def get_p_y(chi, y, N0):
    return np.mean([get_py_on_x(i, y, N0) for i in chi])


def generate_complex_AWGN(x_shape, SNR_db):
    noise_cpx = torch.complex(torch.randn(x_shape), torch.randn(x_shape))
    sigma2 = torch.tensor(1) / hlp.dB2lin(SNR_db, 'dB')  # 1 corresponds to the Power
    noise = torch.sqrt(sigma2) * torch.rsqrt(torch.tensor(2)) * noise_cpx
    noise_power = torch.mean(torch.square(torch.abs(noise)))
    return noise, sigma2, noise_power


def SNRtoMI(N, effSNR, constellation):
    N = int(N)

    SNRlin = 10 ** (effSNR / 10)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    M = constellation.size

    ## Simulation
    x_id = np.random.randint(0, M, (N,))
    x = constellation[:, x_id]

    z = 1 / np.sqrt(2) * (np.random.normal(size=x.shape) + 1j * np.random.normal(size=x.shape));
    y = x + z * np.sqrt(1 / SNRlin);

    return calcMI_MC(x, y, constellation)


def gaussianMI_ASK_Non_Uniform(idx, x, y, constellation, M, P_X, dtype=torch.double):
    """
        Computes mutual information with Gaussian auxiliary channel assumption and constellation with given porbability distribution
        x: (1, N), N normalized complex samples at the transmitter, where N is the batchSize/sampleSize
        y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
        constellation: (1, M), normalized complex constellation of order M
        P_X: (1, M), probability distribution
    """
    if len(constellation.shape) == 1:
        constellation = torch.unsqueeze(constellation, dim=0)
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, dim=0)
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, dim=0)
    if len(idx.shape) == 1:
        idx = torch.unsqueeze(idx, dim=0)
    if len(P_X.shape) == 1:
        P_X = torch.unsqueeze(P_X, dim=0)
    if y.shape[0] != 1:
        y = torch.transpose(y, dim0=0, dim1=1)
    if x.shape[0] != 1:
        x = torch.transpose(x, dim0=0, dim1=1)
    if constellation.shape[0] == 1:
        constellation = torch.transpose(constellation, dim0=0, dim1=1)
    if P_X.shape[0] == 1:
        P_X = torch.transpose(P_X, dim0=0, dim1=1)
    if idx.shape[0] == 1:
        idx = torch.transpose(idx, dim0=0, dim1=1)

    PI = torch.pi
    REALMIN = torch.tensor(np.finfo(float).tiny, dtype=dtype)

    N0 = torch.mean(torch.square(torch.abs(x - y)))

    qY = []

    qYonX = (1 / torch.sqrt(torch.tensor(2) * torch.pi * N0)) * torch.exp( -torch.square(y - x) / (torch.tensor(2) * N0))

    for ii in np.arange(M):
        temp = P_X[ii] * (1 / torch.sqrt(torch.tensor(2) * torch.pi * N0)) * torch.exp(
            -torch.square(y - constellation[ii]) / (torch.tensor(2) * N0))
        qY.append(temp)

    qY = torch.sum(torch.cat(qY, dim=0), dim=0)

    qXonY = P_X[idx] * torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)

    HX = -p_norm(P_X, P_X, lambda x: torch.log2(x))

    MI = HX - torch.mean(-torch.log2(qXonY))

    return MI

def gaussianMI_Non_Uniform(idx, x, y, constellation, M, P_X, dtype=torch.double):
    """
        Computes mutual information with Gaussian auxiliary channel assumption and constellation with given porbability distribution
        x: (1, N), N normalized complex samples at the transmitter, where N is the batchSize/sampleSize
        y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
        constellation: (1, M), normalized complex constellation of order M
        P_X: (1, M), probability distribution
    """
    if len(constellation.shape) == 1:
        constellation = torch.unsqueeze(constellation, dim=0)
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, dim=0)
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, dim=0)
    if len(idx.shape) == 1:
        idx = torch.unsqueeze(idx, dim=0)
    if len(P_X.shape) == 1:
        P_X = torch.unsqueeze(P_X, dim=0)
    if y.shape[0] != 1:
        y = torch.transpose(y, dim0=0, dim1=1)
    if x.shape[0] != 1:
        x = torch.transpose(x, dim0=0, dim1=1)
    if constellation.shape[0] == 1:
        constellation = torch.transpose(constellation, dim0=0, dim1=1)
    if P_X.shape[0] == 1:
        P_X = torch.transpose(P_X, dim0=0, dim1=1)
    if idx.shape[0] == 1:
        idx = torch.transpose(idx, dim0=0, dim1=1)

    N = torch.tensor(list(x.shape)[1], dtype=dtype)

    PI = torch.pi
    REALMIN = torch.tensor(np.finfo(float).tiny, dtype=dtype)

    N0 = torch.mean(torch.square(torch.abs(x - y)))

    qYonX = 1 / (PI * N0) * torch.exp(
        (-torch.square(y.real - x.real) - torch.square(y.imag - x.imag)) / N0)

    qY = []

    for ii in np.arange(M):
        temp = P_X[ii] * (1 / (PI * N0) * torch.exp((-torch.square(
            y.real - constellation[ii, 0].real) - torch.square(
            y.imag - constellation[ii, 0].imag)) / N0))
        qY.append(temp)

    qY = torch.sum(torch.cat(qY, dim=0), dim=0)

    qXonY = P_X[idx] * torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)

    HX = -p_norm(P_X, P_X, lambda x: torch.log2(x))

    MI = HX + torch.mean(torch.log2(qXonY))

    return MI


def gaussianMI(x, y, constellation, M, dtype=torch.double):
    """
        Computes mutual information with Gaussian auxiliary channel assumption and constellation with uniform porbability distribution
        x: (1, N), N normalized complex samples at the transmitter, where N is the batchSize/sampleSize
        y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
        constellation: (1, M), normalized complex constellation of order M

        Transcribed from Dr. Tobias Fehenberger MATLAB code.
        See: https://www.fehenberger.de/#sourcecode
    """
    if len(constellation.shape) == 1:
        constellation = torch.unsqueeze(constellation, dim=0)
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, dim=0)
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, dim=0)
    if y.shape[0] != 1:
        y = torch.transpose(y, dim0=0, dim1=1)
    if x.shape[0] != 1:
        x = torch.transpose(x, dim0=0, dim1=1)
    if constellation.shape[0] == 1:
        constellation = torch.transpose(constellation, dim0=0, dim1=1)

    N = torch.tensor(list(x.shape)[1], dtype=dtype)

    PI = torch.pi
    REALMIN = torch.tensor(np.finfo(float).tiny, dtype=dtype)

    xint = torch.argmin(torch.square(torch.abs(x - constellation)), axis=0).type(torch.int32)
    x_count = torch.bincount(xint)
    x_count = torch.reshape(x_count, (M,))
    P_X = x_count.type(torch.float) / N

    N0 = torch.mean(torch.square(torch.abs(x - y)))

    qYonX = 1 / (PI * N0) * torch.exp(
        (-torch.square(y.real - x.real) - torch.square(y.imag - x.imag)) / N0)

    qY = []
    for ii in np.arange(M):
        temp = P_X[ii] * (1 / (PI * N0) * torch.exp((-torch.square(
            y.real - constellation[ii, 0].real) - torch.square(
            y.imag - constellation[ii, 0].imag)) / N0))
        qY.append(temp)
    qY = torch.sum(torch.cat(qY, dim=0), dim=0)

    MI = 1 / N * torch.sum(torch.log2(torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)))

    return MI


def calcMI_MC(x, y, constellation):
    """
        Transcribed from Dr. Tobias Fehenberger MATLAB code.
        See: https://www.fehenberger.de/#sourcecode
    """
    if y.shape[0] != 1:
        y = y.T
    if x.shape[0] != 1:
        x = x.T
    if constellation.shape[0] == 1:
        constellation = constellation.T

    M = constellation.size
    N = x.size
    P_X = np.zeros((M, 1))

    x = x / np.sqrt(np.mean(np.abs(x) ** 2))  # normalize such that var(X)=1
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))  # normalize such that var(Y)=1

    ## Get X in Integer Representation
    xint = np.argmin(np.abs(x - constellation) ** 2, axis=0)

    fun = lambda h: np.dot(h * x - y, np.conj(h * x - y).T)
    h = fminbound(fun, 0, 2)
    N0 = np.real((1 - h ** 2) / h ** 2)
    y = y / h

    ## Find constellation and empirical input distribution
    for s in np.arange(M):
        P_X[s] = np.sum(xint == s) / N

    ## Monte Carlo estimation of (a lower bound to) the mutual information I(XY)
    qYonX = 1 / (np.pi * N0) * np.exp((-(np.real(y) - np.real(x)) ** 2 - (np.imag(y) - np.imag(x)) ** 2) / N0)

    qY = 0
    for ii in np.arange(M):
        qY = qY + P_X[ii] * (1 / (np.pi * N0) * np.exp((-(np.real(y) - np.real(constellation[ii, 0])) ** 2 - (
                np.imag(y) - np.imag(constellation[ii, 0])) ** 2) / N0))

    realmin = np.finfo(float).tiny
    MI = 1 / N * np.sum(np.log2(np.maximum(qYonX, realmin) / np.maximum(qY, realmin)))

    return MI


def generateBitVectors(N, M):
    # Generates N bit vectors with M bits
    w = int(np.log2(M))
    d = np.zeros((N, w))
    r = np.random.randint(low=0, high=M, size=(N,))
    for ii in range(N):
        d[ii, :] = np.array([float(x) for x in np.binary_repr(r[ii], width=w)])
    return d


def generateUniqueBitVectors(M):
    # Generates log2(M) unique bit vectors with M bits
    w = int(np.log2(M))
    d = np.zeros((M, w))
    for ii in range(M):
        d[ii, :] = np.array([float(x) for x in np.binary_repr(ii, width=w)])
    return d


def py_BPSK(y, P, N0):
    return 0.5 * np.power(2 * np.pi * N0, -0.5) * \
           (np.exp(-np.square(y + np.sqrt(P)) / (2 * N0)) + np.exp(-np.square(y - np.sqrt(P)) / (2 * N0)))


def cap_BPSK(P, N0):
    py = [py_BPSK(y, P, N0) for y in np.arange(-1000, 1000, 1)]
    return scipy.stats.entropy(py, base=2) - 0.5 * np.log2(2 * np.pi * np.e * N0)

def plot_references():
    fig2, ax2 = plt.subplots()
    plot_snr = np.arange(-5, 30, 1)
    AWGN_cap = [np.log2(1 + hlp.dB2lin(p, 'dB')) for p in plot_snr]
    BPSK_cap = [cap_BPSK(P=hlp.dB2lin(p, 'dB'), N0=1) for p in plot_snr]
    QPSK_cap = [2 * cap_BPSK(P=hlp.dB2lin(p, 'dB'), N0=1) for p in plot_snr]
    # QAM_16 = [4 * cap_BPSK(P=hlp.dB2lin(p, 'dB'), N0=1) for p in plot_snr]
    # QAM_64 = [8 * cap_BPSK(P=hlp.dB2lin(p, 'dB'), N0=1) for p in plot_snr]

    ax2.plot(plot_snr, AWGN_cap, label='AWGN')
    ax2.plot(plot_snr, BPSK_cap, label='BPSK')
    ax2.plot(plot_snr+2, QPSK_cap, label='QPSK')
    # ax2.plot(plot_snr, QAM_16, label='16-QAM')
    # ax2.plot(plot_snr, QAM_64, label='64-QAM')
    return fig2, ax2
