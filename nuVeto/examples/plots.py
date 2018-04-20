import pickle
import os
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import *
from matplotlib import pyplot as plt
from scipy import interpolate
import pandas as pd
try:
    import CRFluxModels.CRFluxModels as pm
except ImportError:
    import CRFluxModels as pm


# passing fraction tests
def fn(slice_val):
    """ decide which fn to run depending on slice_val
    """
    return pr_enu if slice_val <=1 else pr_cth


def pr_enu(cos_theta=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, accuracy=3, fraction=True, prpl='step_1', corr_only=False, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(3,7,100) if corr_only else np.logspace(3,7,20)
    passed = [passing(en, cos_theta, kind, pmodel, hadr, barr_mods, depth, accuracy, fraction, prpl, corr_only) for en in ens]
    passed_fn = interpolate.interp1d(ens, passed, kind='quadratic')
    ens_plot = np.logspace(3,7,100)
    if fraction:
        prs = plt.plot(ens_plot, passed_fn(ens_plot), **kwargs)
        plt.ylim(-0.05, 1.05)
        plt.ylabel(r'Passing fraction')
    else:
        prs = plt.plot(ens_plot, passed_fn(ens_plot)*ens_plot**3, **kwargs)
        plt.yscale('log')
        plt.ylabel(r'$E_\nu^3 \Phi_\nu [GeV^2 cm^-2 s^-1 st^-1]$')
    plt.xlim(10**3, 10**7)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$ [GeV]')
    return prs[0]


def pr_cth(enu=1e5, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, accuracy=3, fraction=True, prpl='step_1', corr_only=False, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,21)
    passed = [passing(enu, cos_theta, kind, pmodel, hadr, barr_mods, depth, accuracy, fraction, prpl, corr_only) for cos_theta in cths]
    if fraction:
        prs = plt.plot(cths, passed, **kwargs)
        plt.ylim(-0.05, 1.05)
        plt.ylabel(r'Passing fraction')
    else:
        prs = plt.plot(cths, np.asarray(passed)*enu**3, **kwargs)
        plt.yscale('log')
        plt.ylabel(r'$E_\nu^3 \Phi_\nu [GeV^2 cm^-2 s^-1 st^-1]$')
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    return prs[0]


def depth(slice_val=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', fraction=True):
    depths = np.asarray([1450, 1950, 2450], 'f')*Units.m
    for depth in depths:
        fn(slice_val)(slice_val, kind, pmodel, hadr,
                      depth=depth, fraction=fraction,
                      label='depth {:.0f} m'.format(depth/Units.m))
    plt.title('{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()
        

def brackets(slice_val=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', fraction=True, params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    uppers = [BARR[param].error for param in params]
    lowers = [-BARR[param].error for param in params]
    all_barr_mods = [tuple(zip(params, uppers)), tuple(zip(params, lowers))]
    pr = fn(slice_val)(slice_val, kind, pmodel, hadr, label='{} {:.2g}'.format(kind, slice_val))
    for barr_mods in all_barr_mods:
        fn(slice_val)(slice_val, kind, pmodel, hadr, barr_mods, fraction=fraction,
                      color=pr.get_color(), alpha=1-abs(barr_mods[0][-1]))


def samples(slice_val=1, kind='numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', fraction=True,
                 seed=88, nsamples=10, params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    pr = fn(slice_val)(slice_val, kind, pmodel, hadr=hadr, label='{} {:.2g}'.format(kind, slice_val))
    np.random.seed(seed)
    for i in xrange(nsamples-1):
        # max(-1, throw) prevents throws that dip below -100%
        errors = [max(-1, np.random.normal(scale=BARR[param].error)) for param in params]
        barr_mods = tuple(zip(params, errors))
        fn(slice_val)(slice_val, kind, pmodel, hadr, barr_mods, color=pr.get_color(),
                      alpha=1-min(np.mean(np.abs(errors)), 0.9))


def accuracy(slice_val=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', fraction=True):
    plt.clf()
    accuracies = [2,3,4]
    for accuracy in accuracies:
        fn(slice_val)(slice_val, kind, pmodel=pmodel, hadr=hadr,
                      accuracy=accuracy, fraction=fraction,
                      label='accuracy {}'.format(accuracy))
    plt.title('{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def prpls(slice_val=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', compare=(None, 'step_1', 'sigmoid_0.75_0.1')):
    for prpl in compare:
        fn(slice_val)(slice_val, kind, pmodel=pmodel, hadr=hadr, prpl=prpl,
                      label='{} {} {:.2g}'.format(prpl, kind, slice_val))
    plt.legend()


def elbert(slice_val=1., kind='conv_numu', pmodel=(pm.GaisserHonda, None), prpl='step_1', corr_only=False):
    hadrs=['DPMJET-III', 'SIBYLL2.3', 'SIBYLL2.3c']
    echoice = exthp.corr if corr_only else exthp.passrates
    if slice_val > 1:
        cths = np.linspace(0,1, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(cths))
        plt.plot(cths, echoice(kind)(slice_val, emu, cths), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))
    else:
        ens = np.logspace(2,9, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(slice_val))
        plt.plot(ens, echoice(kind)(ens, emu, slice_val), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))

    for hadr in hadrs:
        fn(slice_val)(slice_val, kind, pmodel, hadr, prpl=prpl, corr_only=corr_only,
                      label='{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def elbert_pmodels(slice_val=1., kind='conv_numu', hadr='SIBYLL2.3c', prpl='step_1', corr_only=False):
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH')]
    echoice = exthp.corr if corr_only else exthp.passrates
    if slice_val > 1:
        cths = np.linspace(0,1, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(cths))
        plt.plot(cths, echoice(kind)(slice_val, emu, cths), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))
    else:
        ens = np.logspace(2,9, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(slice_val))
        plt.plot(ens, echoice(kind)(ens, emu, slice_val), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))
    for pmodel in pmodels:
        pr = fn(slice_val)(slice_val, kind, pmodel[:2], hadr, prpl=prpl, corr_only=corr_only,
                     label='{} {} {:.2g}'.format(pmodel[2], kind, slice_val))
    plt.legend()


def pmodels(slice_val=1., kind='conv_numu', hadr='SIBYLL2.3c', prpl='step_1', fraction=True):
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH'),
               (pm.ZatsepinSokolskaya, 'default', 'ZS')]
    for pmodel in pmodels:
        pr = fn(slice_val)(slice_val, kind, pmodel[:2], hadr, prpl=prpl, fraction=fraction,
                           label='{} {} {:.2g}'.format(pmodel[2], kind, slice_val))
    plt.legend()


def corsika(cos_theta_bin=-1, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3', prpl='step_1', corsika_file='eff_maxmu', plot_nuveto_lines = False, plot_legacy_veto_lines = False):
    if isinstance(cos_theta_bin, list):
        [corsika(cth, kind, pmodel, hadr, prpl, corsika_file) for cth in cos_theta_bin]
        return

    corsika = pickle.load(open(resource_filename('nuVeto', os.path.join('/data/corsika', corsika_file+'.pkl'))))
    fraction = 'eff' in corsika_file
    eff, elow, eup, xedges, yedges = corsika[kind]
    cos_theta = centers(yedges)[cos_theta_bin]

    if plot_legacy_veto_lines and fraction:
        ens = np.logspace(2,9, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(cos_theta))
        plt.plot(ens, exthp.passrates(kind)(ens, emu, cos_theta), 'k--',
                 label='Analytic approx. {} {:.2g}'.format(kind, cos_theta))
    if plot_nuveto_lines:
        pr = pr_enu(cos_theta, kind, pmodel=pmodel, hadr=hadr, prpl=prpl,
                fraction=fraction, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
    plt.errorbar(10**centers(xedges), eff[:,cos_theta_bin],
                 xerr=np.asarray(zip(10**centers(xedges)-10**xedges[:-1],
                                     10**xedges[1:]-10**centers(xedges))).T,
                 yerr=np.asarray(zip(elow[:,cos_theta_bin],
                                     eup[:,cos_theta_bin])).T,
                 label='CORSIKA {} {:.2g}'.format(kind, cos_theta),
                 fmt='.', color=pr.get_color())
    plt.legend()


# intermediate tests
def dndee(mother, daughter):
    sv = SelfVeto(0)
    x_range, dNdEE, dNdEE_interp = sv.get_dNdEE(mother, daughter)

    # print x_range[0], x_range[-1]
    x_samp = np.logspace(1, -9, 5000)
    c = plt.plot(x_samp, dNdEE_interp(x_samp), label = "Interpolated {} to {}".format(mother, daughter))
    plt.plot(x_range,dNdEE, '.', color=c[0].get_color(), label = "MCEq {} to {}".format(mother, daughter))
    plt.semilogx()
    plt.xlabel(r"$x=E_\nu/E_p$")
    plt.ylabel(r"$ \frac{dN}{dE_\nu} E_p$")
    # plt.ylim(-0.1, 5.1)
    plt.axvline(1-ParticleProperties.rr(mother, daughter), linestyle='--', color=c[0].get_color())
    plt.legend()


def plot_prpl(int_prpl, include_mean=False, include_cbar=True):
    plt.scatter(int_prpl[:,0], int_prpl[:,1], c=int_prpl[:,2])
    plt.xlabel(r'$E_\mu^i$ [GeV]')
    plt.ylabel(r'$l_{ice}$ [m]')
    plt.loglog()
    if include_cbar:
        plt.colorbar()
    if include_mean:
        l_ice = np.unique(int_prpl[:,1])
        small_ice = l_ice[l_ice<2.7e4]
        plt.plot(extsv.minimum_muon_energy(small_ice), small_ice, 'k--', label=r'Mean $E_\mu^i$')
        plt.legend()


def parent_flux(cos_theta, parent='D0', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', mag=3,
                     ecr=None, particle=None):
    plt.figure()
    sv = SelfVeto(cos_theta, pmodel, hadr)
    gsol = sv.grid_sol(ecr, particle)
    dh_vec, x_vec = sv.dh_vec, sv.x_vec
    for idx, x_val in enumerate(x_vec):
        mceq = sv.mceq.get_solution(parent, mag, grid_idx=idx)
        calc = sv.get_solution(parent, gsol, mag, grid_idx=idx)
        pout = plt.loglog(sv.mceq.e_grid, mceq,
                          label='h={:.3g} km'.format(
                              float(sv.mceq.density_model.X2h(x_val))/1e5))
        plt.loglog(sv.mceq.e_grid, calc, '--',
                   color=pout[0].get_color())

    plt.xlabel(r'$E_p$')
    plt.ylabel(r'$E_p^{} \Phi_p$'.format(mag))
    plt.ylim(ymin=1e-20)
    plt.legend()
    plt.title('{} {:.2g}'.format(parent, cos_theta))
    # plt.savefig('/Users/tianlu/Desktop/selfveto/parent_flux/combined/{}.png'.format(parent))
        

def nu_flux(cos_theta, kinds='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', mag=3, logxlim=(3,7), corr_only=False):
    sv = SelfVeto(cos_theta, pmodel, hadr)
    sv.grid_sol()
    fig, axs = plt.subplots(2,1)
    for kind in kinds.split():
        plt.sca(axs[0])
        mine = np.asarray([total(en, cos_theta, kind, pmodel, hadr, corr_only=corr_only) for en in sv.mceq.e_grid])
        pr = plt.plot(sv.mceq.e_grid, mine*sv.mceq.e_grid**mag,
                      label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
        plt.ylabel(r'$E_\nu^{} \Phi_\nu$'.format(mag))
        plt.loglog()
        plt.xlim(*np.power(10,logxlim))
        plt.ylim(ymin=1e-8)
        plt.legend()

        try:
            theirs = sv.mceq.get_solution(kind)
            pr = plt.plot(sv.mceq.e_grid, theirs*sv.mceq.e_grid**mag,
                          linestyle='--', color=pr[0].get_color())

            plt.sca(axs[1])
            plt.plot(sv.mceq.e_grid, theirs/mine)
        except KeyError:
            plt.sca(axs[1])
        finally:
            plt.ylabel(r'ratio MCEq/Calc')
            plt.xscale('log')
            plt.ylim(0.5, 1.9)
            plt.xlabel(r'$E_\nu$ [GeV]')
            plt.xlim(*np.power(10,logxlim))


def prob_nomu(cos_theta, particle=14, pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', prpl='step_1'):
    """ plot prob_nomu as fn of ecr
    """
    ecrs = amu(particle)*np.logspace(3, 10, 20)
    ecrs_fine = amu(particle)*np.logspace(3, 10, 1000)
    sv = SelfVeto(cos_theta, pmodel, hadr)
    pnm = [sv.prob_nomu(ecr, particle, prpl) for ecr in ecrs]
    pnmfn = interpolate.interp1d(ecrs, pnm, kind='cubic',
                                 assume_sorted=True, fill_value=(1,np.nan))
    plt.semilogx(ecrs_fine, pnmfn(ecrs_fine), label='interpolated')
    plt.semilogx(ecrs, pnm, 'ko')
    plt.xlabel(r'$E_{CR} [GeV]$')
    plt.ylabel(r'$e^{-N_\mu}$')
    plt.legend()


def elbert_only(slice_val=1., kind='conv_numu'):
    if 'nue' in kind:
        echoices = [exthp.passrates]
        names = ['uncorr.']
    else:
        echoices = [exthp.corr, exthp.passrates]
        names = ['corr.', 'corr.*uncorr.']
    if slice_val > 1:
        cths = np.linspace(0,1, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(cths))
        for echoice, name in zip(echoices,names):
            plt.plot(cths, echoice(kind)(slice_val, emu, cths), '--', label='{} {} {:.2g}'.format(name, kind, slice_val))
    else:
        ens = np.logspace(2,9, 100)
        emu = extsv.minimum_muon_energy(extsv.overburden(slice_val))
        for echoice, name in zip(echoices,names):
            plt.plot(ens, echoice(kind)(ens, emu, slice_val), '--', label='{} {} {:.2g}'.format(name, kind, slice_val))

    plt.ylim(-0.05, 1.05)
    plt.ylabel(r'Passing fraction')
    plt.xlim(10**3, 10**7)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$ [GeV]')
    plt.legend()


def hist_preach(infile, plotdir=None):
    napf = 36
    df = pd.read_csv(infile, delim_whitespace=True, header=None,
                     names='ei l ef'.split())
    # If the muon doesn't reach, MMC saves ef as -distance traveled
    df[df<0] = 0
    for idx, (ei, sdf) in enumerate(df.groupby('ei')):
        if idx % napf == 0:
            if idx > 0:
                # plt.legend(fontsize=6)
                plt.tight_layout()
                if plotdir is not None:
                    plt.savefig(os.path.join(
                        os.path.expanduser(plotdir), '{}.png'.format((idx-1)/napf)))
            fig, axs = plt.subplots(6, 6, figsize=(10,10))
            # fig.text(0.5, 0.04, r'$E_f$', ha='center', va='center')
            # fig.text(0.06, 0.5, r'$P(E_f|E_i, l)$', ha='center', va='center', rotation='vertical')
            axs = axs.flatten()
        ax = axs[idx%napf]
        ax.set_prop_cycle('color',plt.cm.Blues(np.linspace(0.3,1,100)))
        plt.sca(ax)
        plt.title(r'${:.2g}$ GeV'.format(ei), fontdict={'fontsize':8})
        for l, efs in sdf.groupby('l'):
            bins = calc_bins(efs['ef'])
            counts, edges = np.histogram(efs['ef'], bins=bins, density=True)
            plt.plot(centers(edges), counts, label='{:.3g} km'.format(l/1e3))
        plt.yscale('log')
        # plt.xlim(sdf['ef'].min(), sdf['ef'].max()*1.1)
    # plt.legend(fontsize=6)
    plt.tight_layout()
    if plotdir is not None:
        plt.savefig(os.path.join(
            os.path.expanduser(plotdir), '{}.png'.format((idx-1)/napf)))
