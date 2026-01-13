from pathlib import Path
from collections import namedtuple
import logging
import pickle
import gzip
from importlib import resources
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from .utils import calc_bins, centers, Units

logger = logging.getLogger(__name__)

def hist_preach(infile):
    """ Builds histograms of P_reach based on MMC output text
    """
    Hist = namedtuple('Hist', 'counts edges')
    df = pd.read_csv(infile, sep=r'\s+', header=None,
                     names='ei l ef'.split())
    # If the muon doesn't reach, MMC saves ef as -distance traveled
    df[df < 0] = 0
    preach = []
    for (ei, _l), efs in df.groupby(['ei', 'l']):
        bins = calc_bins(efs['ef'])
        histo = Hist(*np.histogram(efs['ef'], bins=bins, density=True))
        [preach.append((ei, _l, ef, ew, val)) for ef, ew, val in zip(centers(histo.edges),
                                                                    np.ediff1d(
            histo.edges),
            histo.counts)]

    return np.asarray(preach)


def int_ef(preach, plight):
    """ integate p_reach*p_light over e_f to reduce dimensionality for interpolator
    """
    if isinstance(preach, np.ndarray):
        pass
    elif (preach := Path(preach)).is_file():
        if preach.suffix == '.npz':
            preach = np.load(preach)['data']
        elif preach.suffix == '.pklz':
            preach = pickle.load(gzip.open(preach, 'rb'))
        else:
            preach = hist_preach(preach)
    else:
        # search in default directory
        with (resources.files('nuVeto') / 'data' / 'mmc' / f'{preach.stem}.npz').open('rb') as f:
            preach = np.load(f)['data']

    df = pd.DataFrame(preach, columns='ei l ef ew pdf'.split())
    intg = []
    for (ei, _l), sdf in df.groupby(['ei', 'l']):
        intg.append((ei, _l, np.sum(sdf['ew']*sdf['pdf']*plight(sdf['ef']))))

    return np.asarray(intg)


def interp(preach, plight):
    """ returns an interpolate.RegularGridInterpolator of integral preach * plight over e_f, yields prpl
    """
    intg = int_ef(preach, plight)
    df = pd.DataFrame(intg, columns='ei l prpl'.split())
    df = df.pivot_table(index='ei', columns='l', values='prpl').fillna(0)
    return interpolate.RegularGridInterpolator((df.index, df.columns), df.values, bounds_error=False, fill_value=None)


class MuonProb(object):
    def __init__(self, rginterpolator):
        if rginterpolator is None:
            logger.warning('MuonProb initialized with None, median approximation will be used.')
            self.mu_int = self.median_approx
        elif isinstance(rginterpolator, RegularGridInterpolator):
            logger.info('MuonProb initialized with RegularGridInterpolator.')
            self.mu_int = rginterpolator
        elif (fpath := Path(rginterpolator)).is_file():
            logger.info('MuonProb initialized with full path to file.')
            with open(fpath, 'rb') as f:
                if fpath.suffix.lower() == '.npz':
                    self.mu_int = self.load_from_npz(f)
                else:
                    self.mu_int = pickle.load(f)
        else:
            logger.info('MuonProb initialized using existing resources.')
            with (resources.files('nuVeto') / 'data' / 'prpl' / f'{rginterpolator}.npz').open('rb') as f:
                self.mu_int = self.load_from_npz(f)

    @staticmethod
    def load_from_npz(f):
        data = np.load(f)
        ngrid_keys = len([_ for _ in data.keys() if _.startswith('grid_')])
        grid = tuple(data[f'grid_{_}'] for _ in range(ngrid_keys))

        return RegularGridInterpolator(
            grid,
            data['values'],
            method=data['method'].item(),
            fill_value=None if data['fill_value'] == 'None' else data['fill_value'].item(),
            bounds_error=data['bounds_error'].item()
        )

    @staticmethod
    def median_emui(distance):
        """
        Minimum muon energy required to survive the given thickness of ice with at
        least 1 TeV 50% of the time.

        :returns: minimum muon energy [GeV] for 1 TeV
        """
        b, c = 2.52151, 7.13834
        return 1e3 * np.exp(1e-3 * distance / (b) + 1e-8 * (distance**2) / c)

    @staticmethod
    def median_approx(coord):
        coord = np.asarray(coord)
        muon_energy, ice_distance = coord[..., 0], coord[..., 1]
        min_mue = MuonProb.median_emui(ice_distance)*Units.GeV
        return muon_energy > min_mue

    def prpl(self, coord):
        pdets = self.mu_int(coord)
        pdets[pdets > 1] = 1
        return pdets

    @property
    def eis(self):
        """ Returns the values of initial muon energies used to construct the interpolator.
        If median_approx is used (by initializing with None), returns [0., np.inf]
        """
        if isinstance(self.mu_int, RegularGridInterpolator):
            return self.mu_int.grid[0]
        return np.asarray([0., np.inf])

    @property
    def ldists(self):
        """ Returns the values of travel distances used to construct the interpolator.
        If median_approx is used (by initializing with None), returns [0., np.inf]
        """
        if isinstance(self.mu_int, RegularGridInterpolator):
            return self.mu_int.grid[1]
        np.asarray([0., np.inf])
