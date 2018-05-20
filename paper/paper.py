import os
import sys
from pkg_resources import resource_filename
from nuVeto.examples import plots
from nuVeto.resources.mu import mu
from nuVeto.external import selfveto as extsv
from nuVeto.external import helper as exthp
from nuVeto.selfveto import pm, fluxes
from nuVeto.utils import Units
import numpy as np
from scipy.interpolate import interp1d
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


def earth_attenuation(enu, cos_theta, kind='conv_numu'):
    nufate = os.path.expanduser('~/projects/nuFATE/')
    flavor_dict = {'numu':2,'nue':1,'antinue':-1,'antinumu':-2}
    flavor = flavor_dict[kind.split('_')[-1]]
    zenith = np.arccos(cos_theta)

    # nuFATE
    sys.path.append(nufate)
    import cascade as cas
    import earth
    w,v,ci,energy_nodes,phi_0 = cas.get_eigs(flavor,
                                             os.path.join(nufate,
                                                          'data/phiHGextrap.dat'),
                                             os.path.join(nufate,
                                             'data/NuFATECrossSections.h5'))
    sys.path.pop()

    t = earth.get_t_earth(zenith)*Units.Na
    phisol = np.dot(v,(ci*np.exp(w*t)))/phi_0
    phisolfn = interp1d(energy_nodes, phisol, kind='quadratic', assume_sorted=True)
    return phisolfn(enu)

        
def fig_prpl():
    step1000 = resource_filename('nuVeto', 'data/prpl/ice_allm97_step_1.pkl')
    step750 = resource_filename('nuVeto', 'data/prpl/ice_allm97_step_0.75.pkl')
    sigmoid = resource_filename('nuVeto', 'data/prpl/ice_allm97_sigmoid_0.75_0.25.pkl')
    plt.figure()
    plots.plot_prpl(step1000, True, False)
    plt.title(r'Heaviside $\cal P_{\rm light}$')
    plt.tight_layout(0.3)
    plt.savefig('fig/prpl_step1000.png')
    plt.figure()
    plots.plot_prpl(step750, True, False)
    plt.title(r'Heaviside $\cal P_{\rm light}$')
    plt.tight_layout(0.3)
    plt.savefig('fig/prpl_step750.png')
    plt.figure()
    plots.plot_prpl(sigmoid, True, False)
    plt.title(r'Sigmoid $\cal P_{\rm light}$')
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


def fig_prs_ratio():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    labels = [r'ALLM97',
              r'BB']

    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])

        for cos_th in cos_ths:
            clabel = r'$\cos \theta_z = {}$'.format(cos_th)
            allm97 = plots.fn(cos_th)(cos_th, kind, prpl='ice_allm97_step_1',
                                      label=clabel)
            bb = plots.fn(cos_th)(cos_th, kind, prpl='ice_bb_step_1',
                                  linestyle='--',
                                  color=allm97.get_color())
            plt.plot(bb.get_data()[0], allm97.get_data()[1]/bb.get_data()[1], ':', color=allm97.get_color())

        plt.axvline(np.nan, color='k', linestyle='-',
                    label=labels[0])
        plt.axvline(np.nan, color='k', linestyle='--',
                    label=labels[1])
        plt.axvline(np.nan, color='k', linestyle=':',
                    label='ALLM97/BB')
        plt.legend()
        plt.tight_layout(0.3)
        save('fig/prs_ratio_{}.eps'.format(kind))


def fig_prs():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    prpls = ['ice_allm97_step_1', 'ice_bb_step_1']
    labels = [r'ALLM97',
              r'BB']

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
        save('fig/prs_{}.eps'.format(kind))


def fig_pls():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    prpls = ['ice_allm97_step_1', 'ice_allm97_step_0.75', 'ice_allm97_sigmoid_0.75_0.25']
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
        save('fig/pls_{}.eps'.format(kind))


def fig_medium():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    cos_ths = [0.25, 0.85]
    prpls = [('ice_allm97_step_1', 1.95*Units.km),
             ('water_allm97_step_1', 1.95*Units.km),
             ('ice_allm97_step_1', 3.5*Units.km),
             ('water_allm97_step_1', 3.5*Units.km)]
    labels = [r'Ice $1.95$ km',
              r'Water $1.95$ km',
              r'Ice $3.5$ km',
              r'Water $3.5$ km']

    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, prpl in enumerate(prpls):
            for cos_th in cos_ths:
                clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                plots.fn(cos_th)(cos_th, kind, prpl=prpl[0], depth=prpl[1],
                                 label=clabel, linestyle=linestyles[idx])

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label=labels[idx])
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        save('fig/medium_{}.eps'.format(kind))

        
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


def fig_density():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    dmodels = [('CORSIKA',('SouthPole', 'June'), 'MSIS-90-E SP/Jul'),
               ('CORSIKA',('SouthPole', 'December'), 'MSIS-90-E SP/Dec'),
               ('MSIS00',('Karlsruhe', 'July'), 'NRLMSISE-00 KR/Jul'),
               ('MSIS00',('Karlsruhe', 'December'), 'NRLMSISE-00 KR/Dec')]
    cos_ths = [0.25, 0.85]
    for kind in kinds:
        plt.figure()
        plt.title(titling[kind])
        for idx, dmodel in enumerate(dmodels):
            for cos_th in cos_ths:
                clabel = r'$\cos \theta_z = {}$'.format(cos_th) if idx == 0 else None
                plots.fn(cos_th)(cos_th, kind, density=dmodel[:2],
                                 label=clabel, linestyle=linestyles[idx])

            plt.axvline(np.nan, color='k', linestyle=linestyles[idx],
                        label='{}'.format(dmodel[2]))
            plt.gca().set_prop_cycle(None)
        plt.legend()
        plt.tight_layout(0.3)
        save('fig/dmodels_{}.eps'.format(kind, cos_th))


def fig_pmodels():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               # (pm.GlobalSplineFitBeta, None, 'GSF spl'),
               (pm.GaisserStanevTilav, '4-gen', 'GST 4-gen'),
               # (pm.PolyGonato, False, 'poly-gonato'),
               # (pm.GaisserHonda, None, 'GH'),
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


def fig_flux():
    kinds = ['conv_nue', 'pr_nue', 'conv_numu', 'pr_numu']
    ens = [1e4, 1e5]
    cths = np.linspace(0,1,11)
    cths_full = np.concatenate((-cths[:0:-1], cths))
    for enu in ens:
        plt.figure()
        plt.title(r'$E_\nu = {:.0f}$ TeV'.format(enu/1e3))
        plt.axvline(np.nan, color='k', linestyle=':',
                    label='Total')
        plt.axvline(np.nan, color='k',
                    label='Passing')
        for kind in kinds:
            earth = []
            for cth in cths_full:
                earth.append(earth_attenuation(enu, cth, kind))
            earth = np.asarray(earth)

            passing = []
            total = []
            for cth in cths:
                num, den = fluxes(enu, cth, kind)
                passing.append(num)
                total.append(den)
            total = np.asarray(total)
            passing = np.asarray(passing)
            total_full = np.concatenate((total[:0:-1], total))
            passing_full = np.concatenate((total[:0:-1], passing))

            totalfn = interp1d(cths_full, np.log10(earth*total_full*enu**3),
                               kind='quadratic')
            passfn = interp1d(cths_full, np.log10(earth*passing_full*enu**3),
                              kind='quadratic')
            cths_plot = np.linspace(-1,1,100)
            pr = plt.plot(cths_plot, 10**totalfn(cths_plot), ':')
            plt.plot(cths_plot, 10**passfn(cths_plot), color=pr[0].get_color(),
                     label=titling[kind])
        plt.xlabel(r'$\cos \theta_z$')
        plt.ylabel(r'$E_\nu^3 \Phi_\nu$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ st$^{-1}]$')
        plt.xlim(-1,1)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout(0.3)
        save('fig/fluxes_{:.0f}.eps'.format(enu/1e3))
