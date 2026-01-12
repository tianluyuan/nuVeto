#!/usr/bin/env python3
import logging
import argparse
from importlib.resources import as_file, files
from pathlib import Path
import numpy as np
from nuVeto.mu import hist_preach, interp
import pl


def main():
    logger = logging.getLogger('mu.py')
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generate muon detection probability')
    parser.add_argument('mmc', metavar='MMC',
                        help='text file or saved (e.g. npz) histogram containing MMC simulated data (fullpath or filename in data/mmc)')
    parser.add_argument('--plight', default='pl_step_1000',
                        choices=[fn for fn in dir(pl) if fn.startswith('pl_')],
                        help='choice of a plight function to apply as defined in pl.py')
    parser.add_argument('--noconvolution', default=False, action='store_true',
                        help='Generate pdfs of preach from raw MMC output and save to npz')
    parser.add_argument('-o', dest='output', default='mu.npz', type=Path,
                        help='output file. If --noconvolution is False this will be saved into the necessary package directory.')

    args = parser.parse_args()
    if args.noconvolution:
        output = args.output
        hpr = hist_preach(args.mmc)
        np.savez_compressed(output, data=hpr, allow_pickle=False)
    else:
        with as_file(files('nuVeto') / 'data' / 'prpl' / args.output.name) as output:
            if args.output.name != str(args.output):
                logger.warning(f'Overriding {args.output} to {output}. To suppress this warning pass the filename only.')
            intp = interp(args.mmc, getattr(pl, args.plight))
            data_dict = {
                'values': intp.values,
                'method': str(intp.method),
                'fill_value': np.array(str(intp.fill_value) if intp.fill_value is None else intp.fill_value),
                'bounds_error': np.array(intp.bounds_error)
            }

            for i, dim_array in enumerate(intp.grid):
                data_dict[f'grid_{i}'] = dim_array

            np.savez_compressed(output, **data_dict, allow_pickle=False)
    logger.info(f'Output saved into {output}')


if __name__ == '__main__':
    main()
