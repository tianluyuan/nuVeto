import os
import pickle
from MCEq.geometry import EarthGeometry
import numpy as np
from scipy import stats
import ParticleDataTool


class Units(object):
    # units
    Na = 6.0221415e+23 # mol/cm^3
    km = 5.0677309374099995 # km to GeV^-1 value from SQuIDS
    cm = km*1.e-5
    m = km*1.e-3
    gr = 5.62e23 # gr to GeV value from SQuIDS
    sec = 1523000.0 #$ sec to GeV^-1 from SQuIDS
    GeV = 1
    MeV = 1e-3*GeV
    TeV = 1.e3*GeV
    PeV = 1.e3*TeV
    mol_air = 14.5


class ParticleProperties(object):
    modtab = ParticleDataTool.SibyllParticleTable()
    pd = ParticleDataTool.PYTHIAParticleData()
    
    mass_dict = {}; lifetime_dict = {}; pdg_id = {}; sibling = {};

    for k in modtab.part_table:
        pdg_id[k] = modtab.modname2pdg[k]
        mass_dict[k] = pd.mass(pdg_id[k]) * Units.GeV
        lifetime_dict[k] = pd.ctau(pdg_id[k]) * Units.cm

    sibling['numu'] = 'mu+'
    sibling['nue'] = 'e+'
    sibling['antinumu'] = 'mu-'
    sibling['antinue'] = 'e-'

    @staticmethod
    def rr(mother, daughter):
        """ returns ratio of masses
        """
        return (ParticleProperties.mass_dict[ParticleProperties.sibling[daughter]]/ParticleProperties.mass_dict[mother])**2


class MaterialProperties(object):
    a = {}; b = {}; density = {};
    a['ice']=0.249*Units.GeV/Units.m # GeV/mwe
    a['rock']=0.221*Units.GeV/Units.m # GeV/mwe
    b['ice']=0.422e-3/Units.m # 1/mwe
    b['rock']=0.531e-3/Units.m # 1/mwe
    density['ice'] = 0.9167*Units.gr/Units.cm**3 # g/cm^3


class MuonProb(object):
    def __init__(self, pklfile):
        if pklfile is None:
            self.mu_int = self.median_approx
        else:
            self.mu_int = pickle.load(open(os.path.join('data', pklfile+'.pkl')))


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
        muon_energy, ice_distance = coord[:,0], coord[:,1]
        min_mue = self.median_emui(ice_distance)*Units.GeV
        return muon_energy > min_mue

    
    def prpl(self, coord):
        return self.mu_int(coord)
        
    
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
    return 1 if particle==14 else particle/100


def centers(x):
    return (x[:-1]+x[1:])*0.5


def ice_column_density(costh, depth = 1950.*Units.m):
    return (overburden(costh, depth/Units.m, elevation=2400)*Units.m)*MaterialProperties.density["ice"]


def categ_to_mothers(categ, daughter):
    charge = '-' if 'anti' in daughter else '+'
    bar = '-bar' if 'anti' in daughter else ''
    lbar = '-bar' if 'anti' not in daughter else ''
    if categ == 'conv':
        mothers = ['pi'+charge, 'K'+charge, 'K0L']
        if 'nue' in daughter:
            mothers.extend(['K0S'])
    elif categ == 'pr':
        mothers = ['D'+charge, 'Ds'+charge, 'D0'+bar]# 'Lambda0'+lbar, 'LambdaC+'+bar
    else:
        mothers = [categ,]
    return mothers
