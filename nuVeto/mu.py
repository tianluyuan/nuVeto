from pathlib import Path
from collections import namedtuple
import pickle
import gzip
import numpy as np
import pandas as pd
from scipy import interpolate
from importlib import resources
from nuVeto.utils import calc_bins, centers


def hist_preach(infile):
    """ Builds histograms of P_reach based on MMC output text
    """
    Hist = namedtuple('Hist', 'counts edges')
    df = pd.read_csv(infile, sep=r'\s+', header=None,
                     names='ei l ef'.split())
    # If the muon doesn't reach, MMC saves ef as -distance traveled
    df[df < 0] = 0
    preach = []
    for (ei, l), efs in df.groupby(['ei', 'l']):
        bins = calc_bins(efs['ef'])
        histo = Hist(*np.histogram(efs['ef'], bins=bins, density=True))
        [preach.append((ei, l, ef, ew, val)) for ef, ew, val in zip(centers(histo.edges),
                                                                    np.ediff1d(
            histo.edges),
            histo.counts)]

    return np.asarray(preach)


def int_ef(preach, plight):
    """ integate p_reach*p_light over e_f to reduce dimensionality for interpolator
    """
    if Path(preach).is_file():
        try:
            preach = pickle.load(gzip.open(preach, 'rb'), encoding='latin1')
        except IOError:
            preach = hist_preach(preach)
    elif Path(preach).suffix == '.pklz':
        # search in default directory
        preach = resources.files('nuVeto') / 'data' / 'mmc' / preach
        preach = pickle.load(gzip.open(preach, 'rb'), encoding='latin1')

    df = pd.DataFrame(preach, columns='ei l ef ew pdf'.split())
    intg = []
    for (ei, l), sdf in df.groupby(['ei', 'l']):
        intg.append((ei, l, np.sum(sdf['ew']*sdf['pdf']*plight(sdf['ef']))))

    return np.asarray(intg)


def interp(preach, plight):
    intg = int_ef(preach, plight)
    df = pd.DataFrame(intg, columns='ei l prpl'.split())
    df = df.pivot_table(index='ei', columns='l', values='prpl').fillna(0)
    return interpolate.RegularGridInterpolator((df.index, df.columns), df.values, bounds_error=False, fill_value=None)
