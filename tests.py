from mceqveto import *
from matplotlib import pyplot as plt


def test_pr(cos_theta=1., kind='numu', pmods=(), hadr='SIBYLL2.3c', accuracy=20, fraction=True, **kwargs):
    """ plot the passing rate (flux or fraction)
    """
    ens = np.logspace(2,9, 100)
    prs = plt.plot(ens, [passing_rate(en, cos_theta, kind, pmods,
                                      hadr, accuracy) for en in ens],
                   **kwargs)
    plt.xlim(10**3, 10**7)
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.xlabel(r'$E_\nu$')
    plt.ylabel(r'Passing rate')
    return prs[0]


def test_accuracy(cos_theta=1., kind='numu', hadr='SIBYLL2.3c', fraction=True):
    plt.clf()
    accuracies = [5, 9, 17, 33]
    for accuracy in accuracies:
        test_pr(cos_theta, kind, hadr=hadr, accuracy=accuracy, fraction=fraction,
                label='accuracy {}'.format(accuracy))
    plt.title('{} {} cth={}'.format(hadr, kind, cos_theta))
    plt.legend()


def test_brackets(cos_theta=1., kind='numu', hadr='SIBYLL2.3c', fraction=True,
                  params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    uppers = [BARR[param].error for param in params]
    lowers = [-BARR[param].error for param in params]
    all_pmods = [tuple(zip(params, uppers)), tuple(zip(params, lowers))]
    pr = test_pr(cos_theta, kind, hadr=hadr, label='{} cth={}'.format(kind, cos_theta))
    for pmods in all_pmods:
        test_pr(cos_theta, kind, pmods, hadr, fraction=fraction,
                color=pr.get_color(), alpha=1-abs(pmods[0][-1]))


def test_samples(cos_theta=1, kind='numu', hadr='SIBYLL2.3c', fraction=True,
                 seed=88, nsamples=10, params='g h1 h2 i w6 y1 y2 z ch_a ch_b ch_e'):
    params = params.split(' ')
    pr = test_pr(cos_theta, kind, hadr=hadr, fraction=fraction, label='{} cth={}'.format(kind, cos_theta))
    np.random.seed(seed)
    for i in xrange(nsamples-1):
        # max(-1, throw) prevents throws that dip below -100%
        errors = [max(-1, np.random.normal(scale=BARR[param].error)) for param in params]
        pmods = tuple(zip(params, errors))
        test_pr(cos_theta, kind, pmods, hadr, color=pr.get_color(), alpha=1-min(np.mean(np.abs(errors)), 0.9))


def test_hadrs(cos_theta=1, kind='numu', fraction=True):
    hadrs=['SIBYLL2.1', 'QGSJET-II-04', 'EPOS-LHC', 'SIBYLL2.3', 'SIBYLL2.3c']
    for hadr in hadrs:
        pr = test_pr(cos_theta, kind, hadr=hadr, fraction=fraction, label='{} {} cth={}'.format(hadr, kind, cos_theta))


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
    plt.title('{} {} cth={}'.format(particle, kind, cos_theta))
    plt.legend()
    plt.subplot(212)
    plt.loglog()
    plt.xlim(xmax=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$dN/dx$')
