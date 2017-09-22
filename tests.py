import pickle
from external import elbert, selfveto
from uncorrelated_selfveto import *
from utils import centers
from matplotlib import pyplot as plt


def test_fn(slice_val):
    """ decide which fn to run depending on slice_val
    """
    return test_pr if slice_val <=1 else test_pr_cth


def test_pr(cos_theta=1., kind='numu', pmods=(), hadr='SIBYLL2.3c', accuracy=20, fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(2,9, 100)
    prs = plt.plot(ens, [passing_rate(en, cos_theta, kind, pmods,
                                      hadr, accuracy, fraction) for en in ens],
                   **kwargs)
    plt.xlim(10**3, 10**7)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$')
    if fraction:
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_pr_cth(enu=1e5, kind='numu', pmods=(), hadr='SIBYLL2.3c', accuracy=20, fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    cths = np.linspace(0,1,11)
    prs = plt.plot(cths, [passing_rate(enu, cos_theta, kind, pmods,
                                       hadr, accuracy, fraction) for cos_theta in cths],
                   **kwargs)
    plt.xlim(0, 1)
    plt.xscale('linear')
    plt.xlabel(r'$\cos \theta$')
    if fraction:
        plt.ylabel(r'Passing fraction')
    else:
        plt.yscale('log')
        plt.ylabel(r'Passing flux')
    return prs[0]


def test_accuracy(slice_val=1., kind='numu', hadr='SIBYLL2.3c', fraction=True):
    plt.clf()
    accuracies = [5, 9, 17, 33]
    for accuracy in accuracies:
        test_fn(slice_val)(slice_val, kind, hadr=hadr, accuracy=accuracy, fraction=fraction,
                label='accuracy {}'.format(accuracy))
    plt.title('{} {} {:.2g}'.format(hadr, kind, slice_val))
    plt.legend()


def test_brackets(slice_val=1., kind='numu', hadr='SIBYLL2.3c', fraction=True,
                  params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    uppers = [BARR[param].error for param in params]
    lowers = [-BARR[param].error for param in params]
    all_pmods = [tuple(zip(params, uppers)), tuple(zip(params, lowers))]
    pr = test_fn(slice_val)(slice_val, kind, hadr=hadr, label='{} {:.2g}'.format(kind, slice_val))
    for pmods in all_pmods:
        test_fn(slice_val)(slice_val, kind, pmods, hadr, fraction=fraction,
                color=pr.get_color(), alpha=1-abs(pmods[0][-1]))


def test_samples(slice_val=1, kind='numu', hadr='SIBYLL2.3c', fraction=True,
                 seed=88, nsamples=10, params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    pr = test_fn(slice_val)(slice_val, kind, hadr=hadr, fraction=fraction, label='{} {:.2g}'.format(kind, slice_val))
    np.random.seed(seed)
    for i in xrange(nsamples-1):
        # max(-1, throw) prevents throws that dip below -100%
        errors = [max(-1, np.random.normal(scale=BARR[param].error)) for param in params]
        pmods = tuple(zip(params, errors))
        test_fn(slice_val)(slice_val, kind, pmods, hadr, color=pr.get_color(), alpha=1-min(np.mean(np.abs(errors)), 0.9))


def test_hadrs(slice_val=1, kind='numu', fraction=True):
    hadrs=['DPMJET-III', 'QGSJET-II-04', 'EPOS-LHC', 'SIBYLL2.3', 'SIBYLL2.3c']
    for hadr in hadrs:
        pr = test_fn(slice_val)(slice_val, kind, hadr=hadr, fraction=fraction, label='{} {} {:.2g}'.format(hadr, kind, slice_val))


def test_elbert(cos_theta=1, kind='pr_nue'):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    ens = np.logspace(2,9, 100)
    emu = selfveto.overburden(cos_theta)
    plt.plot(ens, elbert.uncorr(kind)(ens, emu, cos_theta), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, cos_theta))
    for hadr in hadrs:
        pr = test_pr(cos_theta, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, cos_theta))


def test_elbert_cth(enu=1e5, kind='pr_nue'):
    hadrs=['DPMJET-III', 'SIBYLL2.3c']
    cths = np.linspace(0,1, 100)
    emu = selfveto.overburden(cths)
    plt.plot(cths, elbert.uncorr(kind)(enu, emu, cths), 'k--', label='Elbert approx. {} {:.2g}'.format(kind, enu))
    for hadr in hadrs:
        pr = test_pr_cth(enu, kind, hadr=hadr, fraction=True, label='{} {} {:.2g}'.format(hadr, kind, enu))


def test_corsika(cos_theta_bin=-1, kind='pr_nue', hadr='SIBYLL2.3'):
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


def test_yields(cos_theta=1, particle=14, kind='mu', pmods=(), hadr='SIBYLL2.3c', **kwargs):
    plt.clf()
    eps = amu(particle)*np.logspace(2, 10, 9)
    for ep in eps:
        sols = mceq_yield(ep, cos_theta, particle, kind, pmods, hadr)
        plt.subplot(211)
        plt.plot(sols.info.e_grid,
                 sols.yields,
                 label=r'{:.2g} GeV'.format(ep))
        plt.subplot(212)
        plt.plot(amu(particle)*sols.info.e_grid/ep,
                 sols.yields*ep/amu(particle),
                 label=r'{:.2g} GeV'.format(ep))

    plt.subplot(211)
    plt.loglog()
    plt.xlabel(r'$E_l$')
    plt.ylabel(r'$dN/dE_l$')
    plt.title('{} {} {:.2g}'.format(particle, kind, cos_theta))
    plt.legend()
    plt.subplot(212)
    plt.loglog()
    plt.xlim(xmax=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$dN/dx$')
