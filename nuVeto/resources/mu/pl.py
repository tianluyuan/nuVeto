import numpy as np
from scipy import stats


def sigmoid(emu, center, scale):
    return stats.norm.cdf(emu, loc=center, scale=scale)


def modified_sigmoid(emu, k0, x0, k1, x1, c):
    x = emu
    return 1 / (1 + np.exp(-k0*(np.log10(x)-x0)))+c*np.exp(-(np.log10(x)-x1)**2/k1)


def pl_step_1000(emu):
    """ heaviside at 1 TeV
    """
    return sigmoid(emu, 1e3, 1e-6)


def pl_step_750(emu):
    """ heaviside at 750 GeV
    """
    return sigmoid(emu, 0.75e3, 1e-6)


def pl_sigmoid_750_100(emu):
    """ sigmoid centered at 750 GeV
    """
    return sigmoid(emu, 750, 100)


def pl_sigmoid_750_250(emu):
    """ sigmoid centered at 750 GeV
    """
    return sigmoid(emu, 750, 250)


def pl_hese(emu):
    return modified_sigmoid(emu, 2.48135679,  3.57243996,  2.05736124,  2.60328639,  0.45801598)*sigmoid(emu, 1., 1e-6)
