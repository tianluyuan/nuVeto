from mceqveto import *
from matplotlib import pyplot as plt


def test_pr(cos_theta=1, kind='numu', mods=()):
    ens = np.logspace(2,9, 100)
    plt.plot(ens, [passing_rate(en, cos_theta, kind=kind, accuracy=20, mods=mods) for en in ens])
    plt.xlim(10**3, 10**7)
    plt.ylim(0, 1)
