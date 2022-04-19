import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

import helper as hlp
import PCS_AE as ae

# Parameters
P = 1  # power
pcs_only = True

# Channel Parameters
chParam = hlp.AttrDict()
chParam.M = 16
chParam.SNR_db = np.array([0, 2, 4, 6, 8, 10])

# Auto-Encoder Parameters
aeParam = hlp.AttrDict()
# aeParam.temperature = 1
# aeParam.nLayersEnc = 1
# aeParam.nLayersDec = 2
# aeParam.nHiddenEnc = 128
# aeParam.nHiddenDec = 128
# aeParam.activation  = tf.nn.relu
# aeParam.dtype       = tf.float32
# aeParam.cpx_dtype   = tf.complex64

# Training Parameters
trainingParam = hlp.AttrDict()
# trainingParam.nBatches = 16
trainingParam.batchSize = 1000
trainingParam.learningRate = 0.001
trainingParam.iterations = 4001


def awgn(x, sigma2):
    noise_t = np.sqrt(sigma2)*torch.randn(x.shape)
    return torch.add(x, noise_t)

def normalization(x, P):
    c = torch.mean(x**2)
    return torch.sqrt(P / c) * x


def calculate_py_given_x(z, sig2):
    return (1 / (np.sqrt(2 * np.pi * sig2))) * np.exp(-z ** 2 / sig2 / 2)


# CE loss function and correct with additional term
def custom_loss_fn(ce, logit, ind, zhat, sig2):
    term_1 = ce(logit, ind)
    q = np.amax(nn.functional.softmax(logit, 1).detach().numpy(), 1)  # Q(c_i|Y_n) <-- learned
    p = np.multiply.reduce(calculate_py_given_x(zhat, sig2), 1)  # P(Y_n|c_i)
    term_2 = np.mean(p * np.log2(q))
    return term_1 + term_2


def complex2real(cplx):
    real = torch.flatten(cplx.real)
    imag = torch.flatten(cplx.imag)
    result = torch.transpose(torch.stack((real, imag)), 0, 1)
    return result


def real2complex(data):
    return data[...,0] + 1j * data[...,1]


def plot_2D_PDF(const, pmf):
    s = pmf * 3000
    plt.scatter(const.real, const.imag, s, c="r")
    plt.grid()
    plt.show()


# Initialize network
dist_generator = ae.DistGenerator(chParam.M)
mapper = ae.Mapper(chParam.M)
demap = ae.Demapper(chParam.M)
loss_fn = nn.CrossEntropyLoss()

for (k, snr) in enumerate(hlp.db_to_lineal(chParam.SNR_db)):
    sigma2 = P / snr
    print(f'---SNR = {chParam.SNR_db[k]} dB---')

    # Optimizer
    if pcs_only:
        optimizer = optim.Adam(list(demap.parameters()) + list(dist_generator.parameters()),
                               lr=trainingParam.learningRate)
    else:
        optimizer = optim.Adam(list(mapper.parameters()) + list(demap.parameters()) + list(dist_generator.parameters()),
                               lr=trainingParam.learningRate)

    # Training loop
    for j in range(trainingParam.iterations):
        # first generate the distribution
        l_M = dist_generator(torch.ones(chParam.M)).reshape(-1, chParam.M)
        P_M = nn.functional.softmax(l_M, 1).detach().numpy()[0]

        # Sample indexes
        indices = hlp.sampler(P_M, trainingParam.batchSize)
        # get onehot from sampled indices
        onehot = np.array([hlp.one_hot(i, chParam.M) for i in indices])
        # convert array to tensors
        onehot_t = torch.tensor(onehot).float()
        indices_t = torch.tensor(indices)
        indices_t = indices_t.type(torch.LongTensor)  # labels

        # Get normalized constellation
        constellation = complex2real(torch.tensor(hlp.get_norm_qam(chParam.M, P, P_M)))

        # If PCS only then the matrix multiplication with the constellation is equivalent
        if (pcs_only):
            xhat = torch.matmul(onehot_t, constellation.float()).float()
        else:
            xhat = mapper(onehot_t)

        yhat = normalization(awgn(xhat, sigma2), P)

        l = demap(yhat)

        if True:
            zhat = (yhat - xhat).detach().numpy()
            loss = custom_loss_fn(loss_fn, l, indices_t, zhat, sigma2)
        else:
            loss = loss_fn(l, indices_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Printout and visualization
        if j % 500 == 0:
            print(f'epoch {j}: Loss = {loss.detach().numpy() / np.log(2) :.4f} dB')
        if loss < 1e-3:
            break
    # Data for the plots
    if pcs_only:
        L_M = dist_generator(torch.ones(chParam.M)).reshape(-1, chParam.M)
        PCS = nn.functional.softmax(L_M, 1).detach().numpy()[0]
        constellation = hlp.get_norm_qam(chParam.M, 1, PCS)
        print(PCS)
        print('Should always be one: ', np.sum(PCS))
        plot_2D_PDF(constellation, PCS)
    else:
        a_plot = np.arange(chParam.M)
        onehot_plot = np.array([hlp.one_hot(a_plot[i], chParam.M) for i in range(chParam.M)])
        L_M = dist_generator(torch.ones(chParam.M)).reshape(-1, chParam.M)
        PCS = nn.functional.softmax(L_M, 1).detach().numpy()[0]
        learned_x = mapper(torch.tensor(onehot_plot).float())
        learned_x = normalization(learned_x, P).detach().numpy()
        print(PCS)
        print('Should always be one: ', np.sum(PCS))
        print('Power should be one:', hlp.calculate_avg_power(real2complex(learned_x), PCS))
        plt.scatter(learned_x[:,0], learned_x[:,1], PCS*3000)
        plt.title(f'Learned Constellation for SNR = {chParam.SNR_db[k]} dB')
        plt.grid()
        plt.show()