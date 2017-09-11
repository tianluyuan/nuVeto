from mceqveto import *
from matplotlib import pyplot as plt


def test_pr(cos_theta=1, kind='numu', pmods=(), **kwargs):
    ens = np.logspace(2,9, 100)
    pr = plt.plot(ens, [passing_rate(en, cos_theta, kind=kind, accuracy=20, pmods=pmods) for en in ens], **kwargs)
    plt.xlim(10**3, 10**7)
    plt.ylim(0, 1)
    plt.xscale('log')
    return pr[0]


def test_barr(cos_theta=1, kind='numu'):
    all_pmods = [((211, 'g', 0.3),
                  (211, 'h1', 0.15),(211, 'h2', 0.15),(211, 'i', 0.122),
                  (321, 'w6', 0.4),
                  (321, 'y1', 0.3),(321, 'y2', 0.3),(321, 'z', 0.122)),
                 ((211, 'g', -0.3),
                  (211, 'h1', -0.15),(211, 'h2', -0.15),(211, 'i', -0.122),
                  (321, 'w6', -0.4),
                  (321, 'y1', -0.3),(321, 'y2', -0.3),(321, 'z', -0.122))]
    pr = test_pr(cos_theta, kind, label='{}, cth={}'.format(kind, cos_theta))
    for pmods in all_pmods:
        test_pr(cos_theta, kind, pmods, color=pr.get_color(), alpha=1-abs(pmods[0][-1]))
