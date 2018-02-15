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


def test_pr(cos_theta=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, scale=1e-6, shift=0, prpl=False, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(3,7,50)
    prs = plt.plot(ens, [passing_rate(
        en, cos_theta, kind, pmodel, hadr, accuracy, fraction, scale, shift, prpl) for en in ens], **kwargs)
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


def test_pr_mult(cos_theta=1., kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, scale=1e-6, shift=0, prpl=False, nenu=2, **kwargs):
    """ plot the corr*uncorr passing rate (flux or fraction)
    """
    import uncorrelated_selfveto as usv
    ens = np.logspace(3,7,50)
    corr = np.asarray([passing_rate(en, cos_theta, kind, pmodel,
                                    hadr, accuracy, fraction, scale, shift, prpl) for en in ens])
    uncorr = np.asarray([usv.passing_rate(en, cos_theta, kind.split('_')[1],
                                          hadr=hadr, fraction=fraction) for en in ens])
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


def test_pr_cth(enu=1e5, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, scale=1e-6, shift=0, prpl=False, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,11)
    prs = plt.plot(cths, [passing_rate(
        enu, cos_theta, kind, pmodel, hadr, accuracy, fraction, scale, shift, prpl) for cos_theta in cths], **kwargs)
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    if fraction:
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


def test_preach_scale(cos_theta=1, kind='conv_numu'):
    scales = [1e-6, 0.1, 0.5, 1.]
    for scale in scales:
        test_pr(cos_theta, kind, scale=scale, label=r'$E_\mu^i$ error {:.0%}, {:.2g}'.format(scale, cos_theta))
    plt.legend()


def test_preach_shift(cos_theta=1, kind='conv_numu'):
    shifts = [-0.3, -0.1, 0, 0.1, 0.3]
    for shift in shifts:
        test_pr(cos_theta, kind, shift=shift, label=r'$E_\mu^i$ shifted {:.0%}, {:.2g}'.format(shift, cos_theta))
    plt.legend()


def test_elbert(cos_theta=1, kind='conv_numu', pmodel=(pm.GaisserHonda, None)):
    hadrs=['DPMJET-III']
    ens = np.logspace(2,9, 100)
    emu = jvssv.minimum_muon_energy(jvssv.overburden(cos_theta))
    plt.plot(ens, elbert.corr(kind)(ens, emu, cos_theta), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, cos_theta))
    for hadr in hadrs:
        pr = test_pr(cos_theta, kind, pmodel=pmodel, hadr=hadr,
                     label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
        test_pr(cos_theta, kind, pmodel=pmodel, hadr=hadr, prpl=True,
                label='Corrected $P_{{reach}}$ {:.2g}'.format(cos_theta), color=pr.get_color(), linestyle='--')
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
        

def test_elbert_cth(enu=1e5, kind='conv_numu', pmodel=(pm.GaisserHonda, None)):
    hadrs=['DPMJET-III']
    cths = np.linspace(0,1, 100)
    emu = jvssv.minimum_muon_energy(jvssv.overburden(cths))
    plt.plot(cths, elbert.corr(kind)(enu, emu, cths), 'k--', label='Analytic approx. {} {:.2g}'.format(kind, enu))
    for hadr in hadrs:
        pr = test_pr_cth(enu, kind, pmodel=pmodel, hadr=hadr, label='{} {} {:.2g}'.format(hadr, kind, enu))
        test_pr_cth(enu, kind, pmodel=pmodel, hadr=hadr, prpl=True, label='{} {} {:.2g}'.format(hadr, kind, enu),
                    color=pr.get_color(), linestyle='--')
    plt.legend()


def test_corsika(cos_theta_bin=-1, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3'):
    if isinstance(cos_theta_bin, list):
        [test_corsika(cth, kind) for cth in cos_theta_bin]
        return

    translate = {'pr_numu':'numu_prompt',
                 'pr_nue':'nue_prompt',
                 'conv_numu':'numu_conv',
                 'conv_nue':'nue_conv'}
    corsika = pickle.load(open('external/corsika/sibyll23.pkl'))
    eff, elow, eup, xedges, yedges = corsika[translate[kind]]
    cos_theta = centers(yedges)[cos_theta_bin]

    pr = test_pr_mult(cos_theta, kind, pmodel=pmodel, hadr=hadr, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
    test_pr_mult(cos_theta, kind, pmodel=pmodel, hadr=hadr, prpl=True, label='Corrected $P_{{reach}}$ {:.2g}'.format(cos_theta), color=pr.get_color(), linestyle='--')
    plt.errorbar(10**centers(xedges), eff[:,cos_theta_bin],
                 xerr=np.asarray(zip(10**centers(xedges)-10**xedges[:-1],
                                     10**xedges[1:]-10**centers(xedges))).T,
                 yerr=np.asarray(zip(elow[:,cos_theta_bin],
                                     eup[:,cos_theta_bin])).T,
                 label='CORSIKA {} {:.2g}'.format(kind, cos_theta),
                 fmt='.', color=pr.get_color())
    plt.legend()


def test_dndee(mother, daughter):
    x_range, dNdEE, dNdEE_interp = get_dNdEE(mother, daughter)

    # print x_range[0], x_range[-1]
    x_samp = np.logspace(1, -9, 5000)
    c = plt.plot(x_samp, dNdEE_interp(x_samp), label = "Interpolated {} to {}".format(mother, daughter))
    plt.plot(x_range,dNdEE, '.', color=c[0].get_color(), label = "MCEq {} to {}".format(mother, daughter))
    plt.semilogx()
    plt.xlabel(r"$x=E_\nu/E_p$")
    plt.ylabel(r"$ \frac{dN}{dE_\nu} E_p$")
    plt.ylim(-0.1, 5.1)
    plt.axvline(1-ParticleProperties.rr(mother, daughter), linestyle='--', color=c[0].get_color())
    plt.legend()


def test_preach(cos_theta=1, scale=0.1, shift=0):
    ice_distance = GEOM.overburden(cos_theta)
    mean = minimum_muon_energy(ice_distance)
    emus = np.linspace(mean-0.5*mean, mean+0.5*mean, 100)
    plt.plot(emus*Units.GeV, muon_reach_prob(emus*Units.GeV, ice_distance, scale, shift),
             label=r'$\cos \theta = {{{}}}$, {:.0%} error, {:.0%} shift'.format(cos_theta, scale, shift))
    plt.xlabel(r'$E_\mu^i [GeV]$')
    plt.ylabel(r'$P_{reach}$')
    plt.legend()


def test_plpr(preach, plight=1e3):
    intg = mu.int_ef(preach, plight)
    plt.scatter(intg[:,0], intg[:,1], c=intg[:,2])
    plt.loglog()
    plt.xlabel(r'$E_i$ [GeV]')
    plt.ylabel(r'Overburden [m]')
    plt.colorbar()
