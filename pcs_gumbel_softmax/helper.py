import numpy as np
import matplotlib.pyplot as plt
import torch

def complex2real(cplx):
    real = cplx.real
    #real = torch.flatten(cplx.real)
    imag = cplx.imag
    #imag = torch.flatten(cplx.imag)
    result = torch.transpose(torch.stack((real, imag)), 0, 1)
    #result = torch.stack((real, imag))
    return result

def real2complex(real):
    return real[...,0] + 1j * real[...,1]

def lin2dB(lin, dBtype):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10 * torch.log10(lin) - fact


def dB2lin(dB, dBtype):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10 ** ((dB + fact) / 10)