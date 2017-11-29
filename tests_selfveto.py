import pickle
from external import elbert, selfveto
from selfveto import *
from utils import centers
from matplotlib import pyplot as plt


def test_fn(slice_val):
    """ decide which fn to run depending on slice_val
    """
    return test_pr if slice_val <=1 else test_pr_cth


def test_pr(cos_theta=1., kind='numu', hadr='SIBYLL2.3c', fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(3,7,50)
    prs = plt.plot(ens, [passing_rate(
        en, cos_theta, kind, hadr, fraction) for en in ens], **kwargs)
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


def test_pr_cth(enu=1e5, kind='numu', hadr='SIBYLL2.3c', fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,11)
    prs = plt.plot(cths, [passing_rate(
        enu, cos_theta, kind, hadr, fraction) for cos_theta in cths], **kwargs)
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    if fraction:
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_elbert(cos_theta=1, kind='conv_numu'):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    ens = np.logspace(2,9, 100)
    emu = selfveto.minimum_muon_energy(selfveto.overburden(cos_theta))
    plt.plot(ens, elbert.corr(kind)(ens, emu, cos_theta), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, cos_theta))
    for hadr in hadrs:
        pr = test_pr(cos_theta, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))


def test_elbert_cth(enu=1e5, kind='conv_numu'):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    cths = np.linspace(0,1, 100)
    emu = selfveto.minimum_muon_energy(selfveto.overburden(cths))
    plt.plot(cths, elbert.uncorr(kind)(enu, emu, cths), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, enu))
    for hadr in hadrs:
        pr = test_pr_cth(enu, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, enu))
