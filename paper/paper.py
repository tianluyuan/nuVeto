from nuVeto.examples import plots
from nuVeto.resources.mu import mu
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import *
from matplotlib import pyplot as plt


plt.style.use('paper.mplstyle')


def prpl():
    heaviside = mu.int_ef(resource_filename('nuVeto', 'resources/mu/mmc/ice.pklz'), mu.pl.pl_heaviside)
    sigmoid = mu.int_ef(resource_filename('nuVeto', 'resources/mu/mmc/ice.pklz'), mu.pl.pl_heaviside)
    plt.figure()
    plots.plot_prpl(heaviside, include_mean=True)
    plt.figure()
    plots.plot_prpl(heaviside, include_mean=True)
