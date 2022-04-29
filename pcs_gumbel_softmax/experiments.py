import torch
import numpy as np

import tx
import helper as hlp
import utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributions as torch_d

def r2c(x):
    #a = torch.tensor(x, dtype=torch.double)
    return x.type(torch.complex64)

def p_norm(p, x, fun=lambda x: torch.pow(torch.abs(x), 2)):
    return torch.sum(p * fun(x))

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
    x_count = torch.reshape(x_count, (M, ))
    P_X = x_count.type(torch.float) / N

    N0 = torch.mean(torch.square(torch.abs(x - y)))

    qYonX = 1 / (PI * N0) * torch.exp(
        (-torch.square(y.real - x.real) - torch.square(y.imag - x.imag)) / N0)

    qY = []
    for ii in np.arange(M):
        temp = P_X[ii] * (1 / (PI * N0) *torch.exp((-torch.square(
            y.real - constellation[ii, 0].real) - torch.square(
            y.imag - constellation[ii, 0].imag )) / N0))
        qY.append(temp)
    qY = torch.sum(torch.cat(qY, dim=0), dim=0)

    MI = 1 / N * torch.sum(torch.log2(torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)))

    return MI

def gaussianMI_Non_Uniform(x, y, constellation, M, P_X, N0, dtype=torch.double):
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

    N = torch.tensor(list(x.shape)[1], dtype=dtype)

    PI = torch.pi
    REALMIN = torch.tensor(np.finfo(float).tiny, dtype=dtype)

    N0 = torch.mean(torch.square(torch.abs(x - y)))

    qYonX = 1 / (PI * N0) * torch.exp(
        (-torch.square(y.real - x.real) - torch.square(y.imag - x.imag)) / N0)

    qY = []
    for ii in np.arange(M):
        temp = P_X[ii] * (1 / (PI * N0) *torch.exp((-torch.square(
            y.real - constellation[ii, 0].real) - torch.square(
            y.imag - constellation[ii, 0].imag )) / N0))
        qY.append(temp)
    qY = torch.sum(torch.cat(qY, dim=0), dim=0)

    qXonY = P_X * torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)

    HX =  -p_norm(P_X, P_X, lambda x: torch.log2(x))

    MI = HX + torch.mean(torch.log2(qXonY))

    return MI


def calculate_py_given_x(x, y, sigma2):
    return (1/(np.sqrt(2*np.pi*sigma2))) * np.exp(-(y-x)**2/sigma2/2)


def calculate_px_given_y(x, y, sigma2, constellation):
    py = 0
    for i in constellation:
        py += 0.25 * calculate_py_given_x(i, y, sigma2)
    pxy = 0.25 * calculate_py_given_x(x, y, sigma2) / py
    return pxy





SNRdBs = np.arange(0,30,1)
mi = np.zeros(SNRdBs.size)
for idx,snr in enumerate(hlp.dB2lin(SNRdBs, 'dB')):

    s = F.one_hot(torch.tensor(np.random.randint(0, 64, size=(1000)), dtype=torch.int64))
    p_s = torch.tensor(np.ones(64) * 1 / 64)

    constellation = tx.qammod(64)
    constellation_t = torch.tensor(constellation, dtype=torch.cfloat)
    norm_factor = torch.rsqrt(p_norm(p_s, constellation_t))
    norm_constellation = torch.mul(constellation_t, r2c(norm_factor))
    x = torch.matmul(r2c(s), torch.transpose(input=norm_constellation, dim0=0, dim1=1))
    should_always_be_one = p_norm(p_s, norm_constellation)

    # Channel
    noise_cpx = torch.complex(torch.randn(x.shape), torch.randn(x.shape))
    sigma2 = torch.tensor(1) / snr  # 1 corresponds to the Power
    noise_snr = r2c(torch.sqrt(sigma2)) * torch.rsqrt(torch.tensor(2)) * noise_cpx
    # https://stats.stackexchange.com/questions/187491/why-standard-normal-samples-multiplied-by-sd-are-samples-from-a-normal-dist-with

    y = torch.add(x, noise_snr)

    mi[idx] = gaussianMI_Non_Uniform(x, y, norm_constellation, 64, p_s, dtype=torch.double).detach().numpy()
    print("MI: ", mi[idx], "AWGN Cap: ", np.log2(1 + snr))

plt.plot(SNRdBs, mi, label='MI (analytical)')
plt.plot(SNRdBs, np.log2(1+ hlp.dB2lin(SNRdBs, 'dB')), label='AWGN Capacity')
plt.xlabel('SNR [dB]')
plt.ylabel('Mutual Information')
plt.legend()
plt.grid()
plt.show()


