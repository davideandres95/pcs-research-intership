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
chParam.SNR_db = [3, 5, 7, 12, 20, 30]

# Auto-Encoder Parameters
aeParam = utils.AttrDict()
aeParam.temperature = 10
aeParam.nLayersEnc  = 1
aeParam.nLayersDec  = 2
aeParam.nFeaturesEnc  = 128
aeParam.nFeaturesDec  = 128

# Training Parameters
trainingParam = utils.AttrDict()
trainingParam.nBatches      = 16
trainingParam.batchSize     = 32 * chParam.M
trainingParam.learningRate  = 1e-3
trainingParam.iterations    = 30
trainingParam.displayStep   = 5

def p_norm(p, x, fun=lambda x: torch.pow(torch.abs(x), 2)):
    return torch.sum(p * fun(x))

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return F.one_hot(torch.argmax(input, axis=-1), chParam.M)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def plot_2D_PDF(axs, const, pmf, db, k):
    i = k // 2
    j = k % 2
    s = pmf * 400
    axs[i, j].scatter(const.real, const.imag, s, c="r")
    axs[i, j].title.set_text(f'SNR = {db} dB')
    axs[i, j].grid()


def plot_cap(mi, SNR_db):
    fig2, ax2 = utils.plot_references()
    ax2.plot(SNR_db, mi, label='64-QAM-PCS')
    ax2.set_xlabel('SNR [dB]')
    ax2.set_ylabel('Mutual Information')
    ax2.legend()
    ax2.grid()
    fig2.show()


def calculate_avg_power(x, p_s):
    return torch.sum(torch.pow(torch.abs(x), 2) * p_s)


#Constant input
enc_inp = torch.tensor([[1]], dtype=torch.float)
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

mi = np.zeros(len(chParam.SNR_db))

for (k, SNR_db) in enumerate(chParam.SNR_db):
    print(f'---SNR = {chParam.SNR_db[k]} dB---')
    # Initialize network
    encoder = ae.Encoder_Stark(in_features=1, width=aeParam.nFeaturesEnc, out_features=chParam.M)
    decoder = ae.Decoder_Stark(in_features=2, width=aeParam.nFeaturesDec, out_features=chParam.M)
    CEloss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=trainingParam.learningRate)

    # AWGN Capacity
    AWGN_Cap = np.log2(1 + hlp.dB2lin(SNR_db, 'dB'))

    # Training loop
    for j in range(trainingParam.iterations):
        for l in range(trainingParam.nBatches):
            # first generate the distribution
            s_logits = encoder(enc_inp)
            g_dist = torch_d.Gumbel(loc=torch.tensor([0.]), scale=torch.tensor([1.])) # create Gumbel dist
            g = torch.squeeze(g_dist.sample(sample_shape=[trainingParam.batchSize, chParam.M]))
            s_bar = F.softmax(input=((g + s_logits) / aeParam.temperature), dim=1)
            s = STEFunction.apply(s_bar) # straight through estimator

            # normalization & Modulation
            p_s = F.softmax(s_logits, dim=1)
            constellation = tx.qammod(chParam.M)
            constellation_t = torch.tensor(constellation, dtype=torch.cfloat)
            norm_factor = torch.rsqrt(p_norm(p_s, constellation_t))
            norm_constellation = torch.mul(constellation_t, norm_factor)
            x = torch.matmul(s.type(torch.complex64), torch.transpose(input=norm_constellation, dim0=0, dim1=1))
            should_always_be_one = p_norm(p_s, norm_constellation)

            # Channel
            noise_snr, sigma2, noise_power = utils.generate_complex_AWGN(x.shape, SNR_db)
            y = torch.add(x, noise_snr)
            y_power = utils.p_norm(p_s, torch.add(norm_constellation, utils.generate_complex_AWGN(norm_constellation.shape, SNR_db)[0] ))

            # demodulator
            y_vec = hlp.complex2real(torch.squeeze(y))
            dec = decoder(y_vec)

            # loss
            loss = CEloss(dec,  s.type(torch.float).detach())
            entropy_S = -p_norm(p_s, p_s, lambda x: torch.log2(x))
            loss_hat = torch.subtract(loss, entropy_S)

            optimizer.zero_grad()
            loss_hat.backward()

            optimizer.step()

            MI = utils.gaussianMI_Non_Uniform(torch.argmax(s, dim=1), x, y, norm_constellation, chParam.M, p_s, dtype=torch.double).detach().numpy()

        # Printout and visualization
        if j % int(trainingParam.displayStep) == 0:
            print(f'epoch {j}: Loss = {loss_hat.detach().numpy() / np.log(2) :.4f} - always 1: {should_always_be_one :.2} - MI: {MI :.4f} - Cap.: {AWGN_Cap:.4f}')
        if (loss_hat.detach().numpy() / np.log(2)) < -100:
            break

    # Data for the plots
    mi[k] = MI
    p_s_t = F.softmax(encoder(enc_inp), dim=1)
    p_s = p_s_t.detach().numpy()[0]
    constellation = tx.qammod(chParam.M)
    constellation_t = torch.tensor(constellation, dtype=torch.cfloat)
    norm_factor = torch.rsqrt(p_norm(p_s_t, constellation_t))
    norm_constellation = norm_factor * constellation_t
    #print(p_s)
    print('Power should always be one:', p_norm(p_s_t, norm_constellation))
    plot_2D_PDF(axs, constellation, p_s, SNR_db, k)

plot_cap(mi, chParam.SNR_db)
fig.text(.5, .03, trainingParam, ha='center')
plt.savefig(f'../plots/Stark/constellations_{timestr}.png')
fig.show()

