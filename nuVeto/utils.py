import os
import pickle
from importlib import resources
from MCEq.geometry.geometry import EarthGeometry
import mceq_config as config
import numpy as np
from scipy import stats
from particletools.tables import SibyllParticleTable, PYTHIAParticleData


class Units(object):
    # units
    Na = 6.0221415e+23  # mol/cm^3
    km = 5.0677309374099995  # km to GeV^-1 value from SQuIDS
    cm = km*1.e-5
    m = km*1.e-3
    gr = 5.62e23  # gr to GeV value from SQuIDS
    sec = 1523000.0  # $ sec to GeV^-1 from SQuIDS
    day = sec*60**2*24
    yr = day*356.25
    GeV = 1
    MeV = 1e-3*GeV
    TeV = 1.e3*GeV
    PeV = 1.e3*TeV
    mol_air = config.A_target
    phim2 = (m**2*GeV*sec)**-1
    phicm2 = (cm**2*GeV*sec)**-1


class ParticleProperties(object):
    modtab = SibyllParticleTable()
    pd = PYTHIAParticleData()

    mass_dict = {}
    lifetime_dict = {}
    pdg_id = {}
    sibling = {}

    for k in modtab.part_table:
        pdg_id[k] = modtab.modname2pdg[k]
        mass_dict[k] = pd.mass(pdg_id[k]) * Units.GeV
        lifetime_dict[k] = pd.ctau(pdg_id[k]) * Units.cm

    @staticmethod
    def rr(mother, daughter):
        """ returns ratio of masses
        """
        other_masses = []
        mother_pdg = ParticleProperties.pdg_id[mother]
        daughter_pdg = ParticleProperties.pdg_id[daughter]
        for br, prod in ParticleProperties.pd.decay_channels(mother_pdg):
            if daughter_pdg in prod:
                mass_tot = sum([ParticleProperties.pd.mass(abs(prod))
                                for prod in prod])-ParticleProperties.pd.mass(daughter_pdg)
                other_masses.append(mass_tot)

        return (min(other_masses)/ParticleProperties.mass_dict[mother])**2

    @staticmethod
    def br_2body(mother, daughter):
        """ returns the two-body branching ratio if it exists
        """
        mother_pdg = ParticleProperties.pdg_id[mother]
        daughter_pdg = ParticleProperties.pdg_id[daughter]
        brs = 0
        for br, prod in ParticleProperties.pd.decay_channels(mother_pdg):
            if daughter_pdg in prod and len(prod) == 2:
                brs += br
        return brs


class MuonProb(object):
    def __init__(self, pklfile):
        if pklfile is None:
            self.mu_int = self.median_approx
        elif os.path.isfile(pklfile):
            self.mu_int = pickle.load(open(pklfile, 'rb'), encoding='latin1')
        else:
            self.mu_int = pickle.load(open(resources.files(
                'nuVeto') / 'data' / 'prpl' / f'{pklfile}.pkl', 'rb'), encoding='latin1')

    def median_emui(self, distance):
        """
        Minimum muon energy required to survive the given thickness of ice with at
        least 1 TeV 50% of the time.

        :returns: minimum muon energy [GeV] for 1 TeV
        """
        # require that the muon have median energy 1 TeV
        b, c = 2.52151, 7.13834
        return 1e3 * np.exp(1e-3 * distance / (b) + 1e-8 * (distance**2) / c)

    def median_approx(self, coord):
        coord = np.asarray(coord)
        muon_energy, ice_distance = coord[..., 0], coord[..., 1]
        min_mue = self.median_emui(ice_distance)*Units.GeV
        return muon_energy > min_mue

    def prpl(self, coord):
        pdets = self.mu_int(coord)
        pdets[pdets > 1] = 1
        return pdets


class Geometry(EarthGeometry):
    def __init__(self, depth):
        """ Depth of detector and elevation of surface above sea-level
        """
        super(Geometry, self).__init__()
        self.depth = depth
        self.h_obs *= Units.cm
        self.h_atm *= Units.cm
        self.r_E *= Units.cm
        self.r_top = self.r_E + self.h_atm
        self.r_obs = self.r_E + self.h_obs

    def overburden(self, cos_theta):
        """Returns the overburden for a detector at *depth* below some surface
        at *elevation*.

        From law of cosines,
        x^2 == r^2+(r-d)^2-2r(r-d)cos(gamma)
        where
        r*cos(gamma) = r-d+x*cos(theta), solve and return x.

        :param cos_theta: cosine of zenith angle in detector coord
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return (np.sqrt(z**2*cos_theta**2+d*(2*r-d))-z*cos_theta)/Units.m

    def overburden_to_cos_theta(self, l):
        """Returns the theta for a given overburden for a detector at 
        *depth* below some surface at *elevation*.

        From law of cosines,
        x^2 == r^2+(r-d)^2-2r(r-d)cos(gamma)
        where
        r*cos(gamma) = r-d+x*cos(theta), solve and return x.

        :param cos_theta: cosine of zenith angle in detector coord
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return (2*d*r-d**2-l**2)/(2*l*z)

    def cos_theta_eff(self, cos_theta):
        """ Returns the effective cos_theta relative the the normal at earth surface.

        :param cos_theta: cosine of zenith angle (detector)
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return np.sqrt(1-(z/r)**2*(1-cos_theta**2))


def amu(particle):
    """
    :param particle: primary particle's corsika id

    :returns: the atomic mass of particle
    """
    return 1 if particle == 14 else particle/100


def centers(x):
    return (x[:-1]+x[1:])*0.5


def calc_nbins(x):
    ptile = np.percentile(x, 75) - np.percentile(x, 25)
    if ptile == 0:
        return 10
    n = (np.max(x) - np.min(x)) / (2 * len(x)**(-1./3) * ptile)
    return np.floor(n)


def calc_bins(x):
    nbins = calc_nbins(x)
    edges = np.linspace(np.min(x), np.max(x)+2, num=nbins+1)
    # force a bin for 0 efs
    if edges[0] == 0 and edges[1] > 1:
        edges = np.concatenate((np.asarray([0., 1.]), edges[1:]))
    return edges


def mceq_categ_format(kind):
    _c, _d = kind.split()
    if 'bar' in _d:
        _d = f"anti{_d.replace('bar', '')}"
    return f"{_c}_{_d.replace('_', '')}"
