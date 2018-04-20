from nuVeto.examples import plots
from nuVeto.resources.mu import mu
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import *
import matplotlib as mpl
from matplotlib import pyplot as plt


plt.style.use('paper.mplstyle')


def prpl():
    heaviside = mu.int_ef(resource_filename('nuVeto.resources.mu', 'mmc/ice.pklz'), mu.pl.pl_heaviside)
    sigmoid = mu.int_ef(resource_filename('nuVeto.resources.mu', 'mmc/ice.pklz'), mu.pl.pl_smeared)
    plt.figure()
    plots.plot_prpl(heaviside, True, False)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('fig/prpl_heaviside.png')
    plt.figure()
    plots.plot_prpl(sigmoid, True, False)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/prpl_sigmoid.png')

    
def prpl_cbar():
    plt.figure(figsize=(5,1))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(plt.gca(),
                                   norm=norm, orientation='horizontal')
    plt.tight_layout(0.8)
    plt.savefig('fig/prpl_cbar.png')


def compare_prpls():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.3]
    for kind in kinds:
        for cos_th in cos_ths:
            plt.figure()
            plots.prpls(cos_th, kind, compare=('step_1', 'sigmoid_0.75_0.1'))
            plt.legend(['Heaviside', 'Sigmoid'])
            plt.tight_layout()
            plt.savefig('fig/prpls_{}_{}.eps'.format(kind, cos_th))

