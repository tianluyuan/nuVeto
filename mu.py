import os
from collections import namedtuple
import utils
import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt


def calc_nbins(x):
    ptile = np.percentile(x, 75) - np.percentile(x, 25)
    if ptile == 0:
        return 10
    n =  (np.max(x) - np.min(x)) / (2 * len(x)**(-1./3) * ptile)
    return np.floor(n)


def calc_bins(x):
    nbins = calc_nbins(x)
    edges = np.linspace(np.min(x), np.max(x)+2, num=nbins+1)
    # force a bin for 0 efs
    if edges[0] == 0 and edges[1]>1:
        edges = np.concatenate((np.asarray([0., 1.]),edges[1:]))
    return edges


def hist_preach(infile, plotdir=None):
    """ Builds histograms of P_reach based on MMC output text
    """
    Hist = namedtuple('Hist', 'counts edges')
    df = pd.read_csv(infile, delim_whitespace=True, header=None,
                     names='ei l ef'.split())
    # If the muon doesn't reach, MMC saves ef as -distance traveled
    df[df<0] = 0
    preach = []
    napf = 36
    if plotdir is not None:
        for idx, (ei, sdf) in enumerate(df.groupby('ei')):
            if idx % napf == 0:
                if idx > 0:
                    # plt.legend(fontsize=6)
                    plt.tight_layout()
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
                histo = Hist(*np.histogram(efs['ef'], bins=bins, density=True))
                [preach.append((ei, l, ef, ew, val)) for ef,ew,val in zip(utils.centers(histo.edges),
                                                                          np.ediff1d(histo.edges),
                                                                              histo.counts)]
                plt.plot(utils.centers(histo.edges), histo.counts, label='{:.3g} km'.format(l/1e3))
            plt.yscale('log')
            # plt.xlim(sdf['ef'].min(), sdf['ef'].max()*1.1)
        # plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(
            os.path.expanduser(plotdir), '{}.png'.format((idx-1)/napf)))
        plt.close('all')
    else:
        for (ei, l), efs in df.groupby(['ei', 'l']):
            bins = calc_bins(efs['ef'])
            histo = Hist(*np.histogram(efs['ef'], bins=bins, density=True))
            [preach.append((ei, l, ef, ew, val)) for ef,ew,val in zip(utils.centers(histo.edges),
                                                                      np.ediff1d(histo.edges),
                                                                          histo.counts)]

    return np.asarray(preach)


def int_ef(preach, plight=1e3):
    """ integate p_reach*p_light over e_f to reduce dimensionality for interpolator
    """
    df = pd.DataFrame(preach, columns='ei l ef ew pdf'.split())
    intg = []
    for (ei, l), sdf in df.groupby(['ei', 'l']):
        above = sdf.loc[sdf['ef'] > plight]
        intg.append((ei, l, np.sum(above['ew']*above['pdf'])))

    return np.asarray(intg)


def interp(preach, plight=1e3):
    intg = int_ef(preach, plight)
    df = pd.DataFrame(intg, columns='ei l prpl'.split())
    df = df.pivot_table(index='ei', columns='l', values='prpl').fillna(0)
    return interpolate.RegularGridInterpolator((df.index,df.columns), df.values, bounds_error=False, fill_value=0)
