from scipy import stats


def sigmoid(emu, center, scale):
    return stats.norm.cdf(emu, loc=center, scale=scale)


def pl_heaviside(emu):
    """ heaviside at 1 TeV
    """
    return sigmoid(emu, 1e3, 1e-6)


def pl_smeared(emu):
    """ sigmoid centered at 750 GeV
    """
    return sigmoid(emu, 750, 100)
