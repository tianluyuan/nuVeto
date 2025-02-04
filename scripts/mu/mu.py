#!/usr/bin/env python3
import logging
import argparse
import pickle
import gzip
from importlib import resources
from pathlib import Path
from nuVeto.mu import hist_preach, interp
import pl


def main():
    logger = logging.getLogger('mu.py')
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generate muon detection probability')
    parser.add_argument('mmc', metavar='MMC',
                        help='text file or pickled histogram containing MMC simulated data (fullpath or filename in data/mmc)')
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
