import numpy as np
from scipy import stats, interpolate


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


def pl_noearlymu(emu):
    nstrarr = np.fromfile(
        '/Users/tianlu/Projects/icecube/studies/hydrangea/lh/n_str_prob_pspl_full_flat_caus_30_combined.txt', sep=' ')
    nstrarr = nstrarr.reshape(2, nstrarr.shape[0]/2)
    # take P(Nstr|Ee) likelihood from chaack

    def nstr(ee): return 10**interpolate.interp1d(
        np.log10(nstrarr[0]), np.log10(nstrarr[1]),
        kind='linear', bounds_error=False, fill_value='extrapolate')(np.log10(ee))
    return 1-nstr(emu)
