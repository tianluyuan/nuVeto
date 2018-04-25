import os
from pkg_resources import resource_filename
from nuVeto.examples import plots
from nuVeto.resources.mu import mu
from nuVeto.external import selfveto as extsv
from nuVeto.external import helper as exthp
from nuVeto.selfveto import pm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


plt.style.use('paper.mplstyle')

linestyles = ['-', '--', ':', '-.']
titling = {'conv_numu':r'Conventional $\nu_\mu$',
           'conv_nue':r'Conventional $\nu_e$',
           'pr_numu':r'Prompt $\nu_\mu$',
           'pr_nue':r'Prompt $\nu_e$'}


def save(fname):
    try:
        os.makedirs('fig')
        plt.savefig(fname)
    except OSError as e:
        plt.savefig(fname)

    
def fig_prpl():
    heaviside = resource_filename('nuVeto', 'data/prpl/step_1.pkl')
    sigmoid = resource_filename('nuVeto', 'data/prpl/sigmoid_0.75_0.25.pkl')
    plt.figure()
    plots.plot_prpl(heaviside, True, False)
    plt.xlim(1e2, 1e8)
    plt.ylim(1, 40)
    plt.tight_layout(0.3)
    plt.savefig('fig/prpl_heaviside.png')
    plt.figure()
    plots.plot_prpl(sigmoid, True, False)
    plt.xlim(1e2, 1e8)
    plt.ylim(1, 40)
    plt.tight_layout(0.3)
    save('fig/prpl_sigmoid.png')

    
def fig_prpl_cbar():
    plt.figure(figsize=(5,1.5))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(plt.gca(), cmap='magma',
                                   norm=norm, orientation='horizontal')
    cb.set_label(r'${\cal P}_{\rm det}$')
    plt.tight_layout(1)
    save('fig/prpl_cbar.png')


def fig_prpls():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    prpls = ['step_1', 'step_0.75', 'sigmoid_0.75_0.25']
    labels = [r'$\Theta(E_\mu^{\rm f} - 1\,{\rm TeV})$',
              r'$\Theta(E_\mu^{\rm f} - 0.75\,{\rm TeV})$',
              r'$\Phi\left(\frac{E_\mu^{\rm f} - 0.75\,{\rm TeV}}{0.25\,{\rm TeV}}\right)$']
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
        save('fig/prpls_{}.eps'.format(kind))


def fig_hadrs():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    hadrs_prompt = ['SIBYLL2.3c', 'SIBYLL2.3', 'DPMJET-III']
    hadrs_conv = ['SIBYLL2.3c', 'SIBYLL2.3', 'QGSJET-II-04', 'EPOS-LHC']
    cos_ths = [0.25, 0.85]
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        hadrs = hadrs_conv if kind.split('_')[0] == 'conv' else hadrs_prompt
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
        save('fig/hadrs_{}.eps'.format(kind))


def fig_pmodels():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH'),
               (pm.ZatsepinSokolskaya, 'default', 'ZS')]
    cos_ths = [0.25, 0.85]
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
        save('fig/pmodels_{}.eps'.format(kind, cos_th))


def fig_extsv():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    ens = np.logspace(2,9, 100)
    useexts = [False, True]
    labels = ['This work', 'GJKvS']
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, useext in enumerate(useexts):
            for cos_th in cos_ths:
                if useext:
                    emu = extsv.minimum_muon_energy(extsv.overburden(cos_th))
                    plt.plot(ens, exthp.passrates(kind)(ens, emu, cos_th), linestyles[idx])
                else:
                    clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                    plots.fn(cos_th)(cos_th, kind, label=clabel)

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label=labels[idx])
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        save('fig/extsv_{}.eps'.format(kind))
