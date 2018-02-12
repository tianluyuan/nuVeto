import os
from collections import namedtuple
import utils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


HIST = namedtuple('HIST', 'counts edges')

def hist_preach(infile, plotdir=None):
    """ Builds histograms of P_reach based on MMC output text
    """
    df = pd.read_csv(infile, delim_whitespace=True, header=None,
                    names='ei l ef'.split(' '))
    df[df<0] = 0
    preach = {}
    napf = 36
    if plotdir is not None:
        for idx, (ei, sdf) in enumerate(df.groupby('ei')):
            if idx % napf == 0:
                if idx > 0:
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        os.path.expanduser(plotdir), '{}.png'.format((idx-1)/napf)))
                fig, axs = plt.subplots(6, 6, figsize=(10,10))
                # fig.text(0.5, 0.04, r'$E_f$', ha='center', va='center')
                # fig.text(0.06, 0.5, r'$P(E_f|E_i, l)$', ha='center', va='center', rotation='vertical')
                axs = axs.flatten()
            plt.sca(axs[idx%napf])
            plt.title(r'${:.2g}$ GeV'.format(ei), fontdict={'fontsize':8})
            for l, efs in sdf.groupby('l'):
                bins = utils.calc_bins(efs['ef'])
                histo = HIST(*np.histogram(efs['ef'], bins=bins, density=True))
                preach[(ei, l)] = histo
                plt.plot(utils.centers(histo.edges), histo.counts, label='{:.3g} km'.format(l/1e3))
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(
            os.path.expanduser(plotdir), '{}.png'.format((idx-1)/napf)))
        plt.close('all')
    else:
        for vals, efs in df.groupby(['ei', 'l']):
            bins = utils.calc_bins(efs['ef'])
            histo = HIST(*np.histogram(efs['ef'], bins=bins, density=True))
            preach[vals] = histo

    return preach
