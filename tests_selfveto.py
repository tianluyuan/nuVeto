import pickle
from external import elbert
from external import selfveto as jvssv
import mu
from selfveto import *
from utils import *
from matplotlib import pyplot as plt
import CRFluxModels as pm


def test_fn(slice_val):
    """ decide which fn to run depending on slice_val
    """
    return test_pr if slice_val <=1 else test_pr_cth


def test_pr(cos_theta=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, prpl='step_1', **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(3,7,50)
    prs = plt.plot(ens, [passing_rate(
        en, cos_theta, kind, pmodel, hadr, accuracy, fraction, prpl) for en in ens], **kwargs)
    plt.xlim(10**3, 10**7)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$')
    if fraction:
        plt.ylim(-0.05, 1.05)
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_pr_mult(cos_theta=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, prpl='step_1', nenu=0, **kwargs):
    """ plot the corr*uncorr passing rate (flux or fraction)
    """
    import uncorrelated_selfveto as usv
    ens = np.logspace(3,7,50)
    corr = np.asarray([passing_rate(en, cos_theta, kind, pmodel,
                                    hadr, accuracy, fraction, prpl) for en in ens])
    uncorr = np.asarray([usv.passing_rate(en, cos_theta, kind, pmodel=pmodel,
                                          hadr=hadr, fraction=fraction, nenu=nenu, prpl=prpl) for en in ens])
    prs = plt.plot(ens, corr*uncorr, **kwargs)
    plt.xlim(10**3, 10**7)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$')
    if fraction:
        plt.ylim(-0.05, 1.05)
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_pr_cth(enu=1e5, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, prpl='step_1', **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,21)
    prs = plt.plot(cths, [passing_rate(
        enu, cos_theta, kind, pmodel, hadr, accuracy, fraction, prpl) for cos_theta in cths], **kwargs)
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    if fraction:
        plt.ylim(-0.05, 1.05)
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_accuracy(slice_val=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', fraction=True):
    plt.clf()
    accuracies = [2, 3, 4]
    for accuracy in accuracies:
        test_fn(slice_val)(slice_val, kind, pmodel=pmodel, hadr=hadr,
                           accuracy=accuracy, fraction=fraction,
                           label='accuracy {}'.format(accuracy))
    plt.title('{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def test_prpls(slice_val=1., kind='conv_numu', pmodel=(pm.GaisserHonda, None), hadr='SIBYLL2.3c'):
    prpls = [None, 'step_1', 'sigmoid_0.75_0.1']

    ens = np.logspace(2,9, 100)
    for prpl in prpls:
        test_fn(slice_val)(slice_val, kind, pmodel=pmodel, hadr=hadr, prpl=prpl,
                label='{} {} {:.2g}'.format(prpl, kind, slice_val))
    plt.legend()


def test_elbert(slice_val=1., kind='conv_numu', pmodel=(pm.GaisserHonda, None), prpl='step_1'):
    hadrs=['DPMJET-III']
    if slice_val > 1:
        cths = np.linspace(0,1, 100)
        emu = jvssv.minimum_muon_energy(jvssv.overburden(cths))
        plt.plot(cths, elbert.corr(kind)(slice_val, emu, cths), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))
    else:
        ens = np.logspace(2,9, 100)
        emu = jvssv.minimum_muon_energy(jvssv.overburden(slice_val))
        plt.plot(ens, elbert.corr(kind)(ens, emu, slice_val), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, slice_val))

    for hadr in hadrs:
        test_fn(slice_val)(slice_val, kind, pmodel=pmodel, hadr=hadr, prpl=prpl,
                label='{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def test_elbert_pmodels(cos_theta=1, kind='conv_numu', hadr='SIBYLL2.3c'):
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH')]
    ens = np.logspace(2,9, 100)
    emu = jvssv.minimum_muon_energy(jvssv.overburden(cos_theta))
    plt.plot(ens, elbert.corr(kind)(ens, emu, cos_theta), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, cos_theta))
    for pmodel in pmodels:
        pr = test_pr(cos_theta, kind, pmodel=pmodel[:2], hadr=hadr,
                     label='{} {} {:.2g}'.format(pmodel[2], kind, cos_theta))
    plt.legend()


def test_corsika(cos_theta_bin=-1, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3', prpl='step_1', corsika_file='nu_all_maxmu'):
    if isinstance(cos_theta_bin, list):
        [test_corsika(cth, kind) for cth in cos_theta_bin]
        return

    translate = {'pr_numu':'numu_prompt',
                 'pr_nue':'nue_prompt',
                 'conv_numu':'numu_conv',
                 'conv_nue':'nue_conv'}
    corsika = pickle.load(open(os.path.join('external/corsika', corsika_file+'.pkl')))
    eff, elow, eup, xedges, yedges = corsika[translate[kind.replace('anti', '')]]
    cos_theta = centers(yedges)[cos_theta_bin]

    ens = np.logspace(2,9, 100)
    emu = jvssv.minimum_muon_energy(jvssv.overburden(cos_theta))
    plt.plot(ens, elbert.passrates(kind)(ens, emu, cos_theta), 'k--',
             label='Analytic approx. {} {:.2g}'.format(kind, cos_theta))
    pr = test_pr_mult(cos_theta, kind, pmodel=pmodel, hadr=hadr, prpl=prpl, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
    plt.errorbar(10**centers(xedges), eff[:,cos_theta_bin],
                 xerr=np.asarray(zip(10**centers(xedges)-10**xedges[:-1],
                                     10**xedges[1:]-10**centers(xedges))).T,
                 yerr=np.asarray(zip(elow[:,cos_theta_bin],
                                     eup[:,cos_theta_bin])).T,
                 label='CORSIKA {} {:.2g}'.format(kind, cos_theta),
                 fmt='.', color=pr.get_color())
    plt.legend()


def test_pmodels(cos_theta=1, kind='conv_numu', hadr='SIBYLL2.3c'):
    pmodels = [(pm.HillasGaisser2012, 'H3a', 'H3a'),
               (pm.PolyGonato, False, 'poly-gonato'),
               (pm.GaisserHonda, None, 'GH')]
    ens = np.logspace(2,9, 50)
    for pmodel in pmodels:
        pr = test_pr_mult(cos_theta, kind, pmodel=pmodel[:2], hadr=hadr,
                          label='{} {} {:.2g}'.format(pmodel[2], kind, cos_theta))
    plt.legend()
        

def test_dndee(mother, daughter):
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


def test_plot_prpl(int_prpl, include_mean=False):
    plt.scatter(int_prpl[:,0], int_prpl[:,1], c=int_prpl[:,2])
    plt.xlabel(r'$E_\mu^i$ [GeV]')
    plt.ylabel(r'$l_{ice}$ [m]')
    plt.loglog()
    plt.colorbar()
    if include_mean:
        l_ice = np.unique(int_prpl[:,1])
        small_ice = l_ice[l_ice<2.7e4]
        plt.plot(jvssv.minimum_muon_energy(small_ice), small_ice, 'k--', label=r'Mean $E_\mu^i$')
        plt.legend()


def test_parent_flux(cos_theta, parent='D0', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', mag=3):
    plt.figure()
    sv = SelfVeto(cos_theta, pmodel, hadr)
    dh_vec, x_vec = sv.dh_vec, sv.x_vec
    for idx, x_val in enumerate(x_vec):
        mceq = sv.get_solution_orig(parent, mag, grid_idx=idx)
        calc = sv.get_solution(parent, mag, grid_idx=idx)
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
        

def test_nu_flux(cos_theta, kinds='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', mag=3, logxlim=(3,7), direct_mceq=True):
    sv = SelfVeto(cos_theta, pmodel, hadr)
    if direct_mceq:
        fig, axs = plt.subplots(2,1)
        plt.sca(axs[0])
    else:
        plt.figure()
    for kind in kinds.split():
        mine = np.asarray([total_flux(en, cos_theta, kind, pmodel, hadr) for en in sv.mceq.e_grid])
        pr = plt.plot(sv.mceq.e_grid, mine*sv.mceq.e_grid**mag,
                      label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
        plt.ylabel(r'$E_\nu^{} \Phi_\nu$'.format(mag))
        plt.loglog()
        plt.xlim(*np.power(10,logxlim))
        plt.ylim(ymin=1e-8)
        plt.legend()

        if direct_mceq:
            theirs = sv.mceq.get_solution(kind)
            pr = plt.plot(sv.mceq.e_grid, theirs*sv.mceq.e_grid**mag,
                          linestyle='--', color=pr[0].get_color())

            plt.sca(axs[1])
            plt.plot(sv.mceq.e_grid, theirs/mine)
            plt.ylabel(r'ratio MCEq/Calc')
            plt.xscale('log')
            plt.ylim(0.5, 1.9)

        plt.xlabel(r'$E_\nu$')
        plt.xlim(*np.power(10,logxlim))

