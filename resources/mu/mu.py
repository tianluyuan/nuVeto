#!/usr/bin/env python3
from pathlib import Path
from collections import namedtuple
import pickle
import gzip
import argparse
import numpy as np
import pandas as pd
import logging
from scipy import interpolate
from importlib import resources
from nuVeto.utils import calc_bins, centers
import pl


def hist_preach(infile):
    """ Builds histograms of P_reach based on MMC output text
    """
    Hist = namedtuple('Hist', 'counts edges')
    df = pd.read_csv(infile, delim_whitespace=True, header=None,
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
    if isinstance(preach, str) and Path(preach).is_file():
        try:
            preach = pickle.load(gzip.open(preach, 'rb'), encoding='latin1')
        except IOError:
            preach = hist_preach(preach)
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


def main():
    logger = logging.getLogger('mu.py')
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generate muon detection probability')
    parser.add_argument('mmc', metavar='MMC',
                        help='text file or pickled histogram containing MMC simulated data')
    parser.add_argument('--plight', default='pl_step_1000',
                        choices=[fn for fn in dir(pl) if fn.startswith('pl_')],
                        help='choice of a plight function to apply as defined in pl.py')
    parser.add_argument('--noconvolution', default=False, action='store_true',
                        help='Generate pdfs of preach from raw MMC output and save to pklz')
    parser.add_argument('-o', dest='output', default='mu.pkl', type=Path,
                        help='output file. If --noconvolution is False this will be saved into the necessary package directory.')

    args = parser.parse_args()
    if args.noconvolution:
        output = args.output
        hpr = hist_preach(args.mmc)
        pickle.dump(hpr, gzip.open(output, 'wb'), protocol=-1)
    else:
        output = resources.files('nuVeto') / 'data' / 'prpl' / args.output.name
        if args.output.name != str(args.output):
            logger.warning(f'Overriding {args.output} to {output}. To suppress this warning pass the filename only.')
        intp = interp(args.mmc, getattr(pl, args.plight))
        pickle.dump(intp, open(output, 'wb'), protocol=-1)
    logger.info(f'Output pickled into {output}')


if __name__ == '__main__':
    main()
