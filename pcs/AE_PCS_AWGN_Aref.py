# -*- coding: utf-8 -*-:
import time

import matplotlib.pyplot as plt
import torch
import torch.distributions as torch_d
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

import tx
import helper as hlp
import utils
import autoencoder as ae

timestr = time.strftime("%Y%m%d-%H%M%S")

# Parameters
# Channel Parameters
chParam = utils.AttrDict()
chParam.M = 64
chParam.SNR_db = [0, 3, 5, 7, 12, 30]

# Auto-Encoder Parameters
aeParam = utils.AttrDict()
aeParam.temperature = 1
aeParam.nLayersEnc  = 1
aeParam.nLayersDec  = 5
aeParam.nFeaturesEnc  = 128
aeParam.nFeaturesDec  = 128

# Training Parameters
trainingParam = utils.AttrDict()
trainingParam.nBatches      = 32
trainingParam.batchSize     = 7500
trainingParam.learningRate  = 1e-2
trainingParam.iterations    = 31
trainingParam.displayStep   = 5

def sampler(P_M, B):
    samples = torch.empty(0)
    for idx, p in enumerate(P_M):
        occurrences = torch.round(B * p).type(torch.LongTensor)
        samples = torch.cat((samples, torch.ones(occurrences, dtype=torch.int64) * torch.tensor(idx)))
    indexes = torch.randperm(samples.shape[0])
    return samples[indexes]

def calculate_py_given_x(z, sig2):
     return (1 / (torch.sqrt(2 * torch.pi * sig2))) * torch.exp(-torch.square(z) / (sig2 * torch.tensor(2)))

def loss_correction_factor(dec, zhat, sig2):
     q = torch.amax(dec, 1)  # Q(c_i|Y_n) <-- learned
     p = torch.prod(calculate_py_given_x(zhat, sig2/torch.tensor(2)), 1) # P(Y_n|c_i)
     return torch.mean(p * torch.log2(q))

def plot_2D_PDF(axs, const, pmf, db, k):
    i = k // 2
    j = k % 2
    s = pmf * 400
    axs[i, j].scatter(const.real, const.imag, s, c="r")
    axs[i, j].title.set_text(f'SNR = {db} dB')
    axs[i, j].grid()

# Constant input
enc_inp = torch.tensor([[1]], dtype=torch.float)

fig, axs = plt.subplots(3, 2, figsize=(10, 15))


for (k, SNR_db) in enumerate(chParam.SNR_db):
    print(f'---SNR = {chParam.SNR_db[k]} dB---')

    # Initialize network
    encoder = ae.Encoder_Aref(in_features=1, width=aeParam.nFeaturesEnc, out_features=chParam.M)
    decoder = ae.Decoder_Aref(in_features=2, width=aeParam.nFeaturesDec, out_features=chParam.M)
    CEloss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=trainingParam.learningRate)

    # AWGN Capacity
    AWGN_Cap = np.log2(1 + hlp.dB2lin(SNR_db, 'dB'))

    # Training loop
    for j in range(trainingParam.iterations):
        for i in range(trainingParam.nBatches):
            # first generate the distribution
            l_M = encoder(enc_inp)
            P_M = F.softmax(l_M, dim=1)

            # Sample indexes
            indices = sampler(torch.squeeze(P_M), trainingParam.batchSize).type(torch.LongTensor)  # labels
            # get onehot from sampled indices
            onehot = F.one_hot(indices, 64)


            # normalization & Modulation
            constellation = tx.qammod(chParam.M)
            constellation_t = torch.tensor(constellation, dtype=torch.cfloat)
            norm_factor = torch.rsqrt(utils.p_norm(P_M, constellation_t))
            norm_constellation = torch.mul(constellation_t, norm_factor)
            x = torch.matmul(onehot.type(torch.complex64), torch.transpose(input=norm_constellation, dim0=0, dim1=1))
            should_always_be_one = utils.p_norm(P_M, norm_constellation)

            # Channel
            noise_snr, sigma2, noise_power = utils.generate_complex_AWGN(x.shape, SNR_db)
            y = torch.add(x, noise_snr)
            y_power = utils.p_norm(P_M, torch.add(norm_constellation, utils.generate_complex_AWGN(norm_constellation.shape, SNR_db)[0] ))

            # demodulator
            y_vec = hlp.complex2real(torch.squeeze(y))
            dec = decoder(y_vec)

            # loss
            zhat = (y_vec - hlp.complex2real(torch.squeeze(x)))
            #N0 = torch.mean(torch.square(torch.abs(x - y)))
            loss = CEloss(dec, onehot.type(torch.float))
            loss_hat = loss + loss_correction_factor(F.softmax(dec, 1), zhat, sigma2)
            #loss_hat = loss + loss_correction_factor(F.softmax(dec, 1), x, y)

            optimizer.zero_grad()
            loss_hat.backward()
            optimizer.step()

            MI = utils.gaussianMI_Non_Uniform(indices, x, y, norm_constellation, chParam.M, P_M, dtype=torch.double).detach().numpy()

        # Printout and visualization
        if j % int(trainingParam.displayStep) == 0:
            print(f'epoch {j}: Loss = {loss_hat.detach().numpy() / np.log(2) :.4f} - always 1: {should_always_be_one :.2} - MI: {MI :.4f} - Cap.: {AWGN_Cap:.4f}')
        if loss_hat < -100:
            break

    # Data for the plots
    p_s_t = F.softmax(encoder(enc_inp), dim=1)
    p_s = p_s_t.detach().numpy()[0]
    constellation = tx.qammod(chParam.M)
    constellation_t = torch.tensor(constellation, dtype=torch.cfloat)
    norm_factor = torch.rsqrt(utils.p_norm(p_s_t, constellation_t))
    norm_constellation = norm_factor * constellation_t
    #print(p_s)
    print('Power should always be one:', utils.p_norm(p_s_t, norm_constellation))
    plot_2D_PDF(axs, constellation, p_s, SNR_db, k)

fig.text(.5, .03, trainingParam, ha='center')
plt.savefig(f'../plots/Aref/constellations_{timestr}.png')
fig.show()
