import pickle
from external import elbert, selfveto
from selfveto import *
from utils import centers, ParticleProperties
from matplotlib import pyplot as plt


def test_fn(slice_val):
    """ decide which fn to run depending on slice_val
    """
    return test_pr if slice_val <=1 else test_pr_cth


def test_pr(cos_theta=1., kind='conv_numu', hadr='SIBYLL2.3c', accuracy=5, fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(3,7,50)
    prs = plt.plot(ens, [passing_rate(
        en, cos_theta, kind, hadr, accuracy, fraction) for en in ens], **kwargs)
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


def test_pr_cth(enu=1e5, kind='conv_numu', hadr='SIBYLL2.3c', accuracy=5, fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,11)
    prs = plt.plot(cths, [passing_rate(
        enu, cos_theta, kind, hadr, accuracy, fraction) for cos_theta in cths], **kwargs)
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    if fraction:
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_accuracy(slice_val=1., kind='conv_numu', hadr='SIBYLL2.3c', fraction=True):
    plt.clf()
    accuracies = [2, 5, 6]
    for accuracy in accuracies:
        test_fn(slice_val)(slice_val, kind, hadr=hadr,
                           accuracy=accuracy, fraction=fraction,
                           label='accuracy {}'.format(accuracy))
    plt.title('{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def test_elbert(cos_theta=1, kind='conv_numu', accuracy=5):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    ens = np.logspace(2,9, 100)
    emu = selfveto.minimum_muon_energy(selfveto.overburden(cos_theta))
    plt.plot(ens, elbert.corr(kind)(ens, emu, cos_theta), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, cos_theta))
    for hadr in hadrs:
        pr = test_pr(cos_theta, kind, hadr=hadr, accuracy=accuracy,
                     fraction=True, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
    plt.legend()


def test_elbert_cth(enu=1e5, kind='conv_numu'):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    cths = np.linspace(0,1, 100)
    emu = selfveto.minimum_muon_energy(selfveto.overburden(cths))
    plt.plot(cths, elbert.corr(kind)(enu, emu, cths), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, enu))
    for hadr in hadrs:
        pr = test_pr_cth(enu, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, enu))
    plt.legend()


def test_corsika(cos_theta_bin=-1, kind='conv_numu', hadr='SIBYLL2.3'):
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

    pr = test_pr(cos_theta, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))
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
    x_samp = np.logspace(0, -9, 1000)
    c = plt.plot(x_samp, dNdEE_interp(x_samp), label = "Interpolated {} to {}".format(mother, daughter))
    plt.plot(x_range,dNdEE, '.', color=c[0].get_color(), label = "MCEq {} to {}".format(mother, daughter))
    plt.semilogx()
    plt.xlabel(r"$x=E_\nu/E_p$")
    plt.ylabel(r"$ \frac{dN}{dE_\nu} E_p$")
    plt.ylim(-0.1, 5.1)
    plt.axvline(1-ParticleProperties.rr(mother, daughter), linestyle='--', color=c[0].get_color())
    plt.legend()
