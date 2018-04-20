from nuVeto.examples import plots
from nuVeto.resources.mu import mu
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import *
import matplotlib as mpl
from matplotlib import pyplot as plt


plt.style.use('paper.mplstyle')

linestyles = ['-', '--', ':', '-.']
titling = {'conv_numu':r'Conventional $\nu_\mu$',
           'conv_nue':r'Conventional $\nu_e$',
           'pr_numu':r'Prompt $\nu_\mu$',
           'pr_nue':r'Prompt $\nu_e$'}


def prpl():
    heaviside = mu.int_ef(resource_filename('nuVeto.resources.mu', 'mmc/ice.pklz'), mu.pl.pl_heaviside)
    sigmoid = mu.int_ef(resource_filename('nuVeto.resources.mu', 'mmc/ice.pklz'), mu.pl.pl_smeared)
    plt.figure()
    plots.plot_prpl(heaviside, True, False)
    plt.legend()
    plt.xlim(1e2, 1e8)
    plt.ylim(1e3, 2e5)
    plt.tight_layout(0.3)
    plt.savefig('fig/prpl_heaviside.png')
    plt.figure()
    plots.plot_prpl(sigmoid, True, False)
    plt.legend()
    plt.xlim(1e2, 1e8)
    plt.ylim(1e3, 2e5)
    plt.tight_layout(0.3)
    plt.savefig('fig/prpl_sigmoid.png')

    
def prpl_cbar():
    plt.figure(figsize=(5,1.5))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(plt.gca(),
                                   norm=norm, orientation='horizontal')
    cb.set_label(r'${\cal P}_{\rm det}$')
    plt.tight_layout(1)
    plt.savefig('fig/prpl_cbar.png')


def compare_prpls():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.3, 0.5]
    prpls = ['step_1', 'sigmoid_0.75_0.3']
    labels = ['Heaviside', 'Sigmoid']
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, prpl in enumerate(prpls):
            for cos_th in cos_ths:
                clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                plots.fn(cos_th)(cos_th, kind, prpl=prpl,
                                 label=clabel, linestyle=linestyles[idx])

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label=labels[idx])
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        plt.savefig('fig/prpls_{}.eps'.format(kind))


def compare_hadrs():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    hadrs=['DPMJET-III', 'SIBYLL2.3', 'SIBYLL2.3c']
    cos_ths = [0.3]
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, hadr in enumerate(hadrs):
            for cos_th in cos_ths:
                clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                plots.fn(cos_th)(cos_th, kind, hadr=hadr,
                                 label=clabel, linestyle=linestyles[idx])

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label=hadr)
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        plt.savefig('fig/hadrs_{}.eps'.format(kind))


def compare_pmodels():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH'),
               (pm.ZatsepinSokolskaya, 'default', 'ZS')]
    cos_ths = [0.3]
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, pmodel in enumerate(pmodels):
            for cos_th in cos_ths:
                clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                plots.fn(cos_th)(cos_th, kind, pmodel=pmodel[:-1],
                                 label=clabel, linestyle=linestyles[idx])

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label=pmodel[-1])
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        plt.savefig('fig/pmodels_{}.eps'.format(kind, cos_th))
