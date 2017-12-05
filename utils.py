from MCEq.geometry import EarthGeometry
import numpy as np
from scipy import stats


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


class ParticleProperties(object):
    mass_dict = {}; lifetime_dict = {}; pdg_id = {}; sibling = {};

    mass_dict['e+']=0.510998*Units.MeV
    mass_dict['mu+']=0.1056583745*Units.GeV
    mass_dict['e-']=0.510998*Units.MeV
    mass_dict['mu-']=0.1056583745*Units.GeV
    mass_dict['K+']=0.493677*Units.GeV # GeV
    mass_dict['K0L']=0.497611*Units.GeV # GeV
    mass_dict['pi+']=0.139570*Units.GeV # GeV
    mass_dict['K-']=0.493677*Units.GeV # GeV
    mass_dict['pi-']=0.139570*Units.GeV # GeV
    mass_dict['D+']=1.86962*Units.GeV # GeV
    mass_dict['D-']=1.86962*Units.GeV # GeV
    mass_dict['Ds+']=1.96830*Units.GeV # GeV
    mass_dict['Ds-']=1.96830*Units.GeV # GeV
    mass_dict['D0']=1.86484*Units.GeV # GeV
    mass_dict['D0-bar']=1.86484*Units.GeV # GeV
    mass_dict['air']=(14.5)*Units.GeV # GeV

    lifetime_dict['K+']=1.2389e-8*Units.sec # s
    lifetime_dict['K0L']=5.116e-8*Units.sec # s
    lifetime_dict['pi+']=2.6033e-8*Units.sec # 
    lifetime_dict['K-']=1.2389e-8*Units.sec # s
    lifetime_dict['pi-']=2.6033e-8*Units.sec # 
    lifetime_dict['D+']=1.040e-12*Units.sec # seconds to usual Units
    lifetime_dict['D-']=1.040e-12*Units.sec # seconds to usual Units
    lifetime_dict['Ds+']=5.00e-13*Units.sec
    lifetime_dict['Ds-']=5.00e-13*Units.sec
    lifetime_dict['D0']=4.101*Units.sec
    lifetime_dict['D0-bar']=4.101*Units.sec

    pdg_id['D+'] = 411 # D+
    pdg_id['K+'] = 321 # k+
    pdg_id['K0S'] = 310
    pdg_id['K0L'] = 130
    pdg_id['pi+'] = 211 # pi+
    pdg_id['D-'] = -411 # D+
    pdg_id['K-'] = -321 # k+
    pdg_id['pi-'] = -211 # pi+
    pdg_id['Ds+'] = 431
    pdg_id['Ds-'] = -431
    pdg_id['D0'] = 421
    pdg_id['D0-bar'] = -421
    pdg_id['numu'] = 14
    pdg_id['nue'] = 12
    pdg_id['antinumu'] = -14
    pdg_id['antinue'] = -12

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


def minimum_muon_energy(distance):
    """
    Minimum muon energy required to survive the given thickness of ice with at
    least 1 TeV 50% of the time.

    :returns: minimum muon energy [GeV]
    """
    # require that the muon have median energy 1 TeV
    b, c = 2.52151, 7.13834
    return 1e3 * np.exp(1e-3 * distance / (b) + 1e-8 * (distance**2) / c)


def ice_column_density(costh, depth = 1950.*Units.m):
    return (overburden(costh, depth/Units.m, elevation=2400)*Units.m)*MaterialProperties.density["ice"]


def muon_reach_prob(muon_energy, ice_distance, scale=0.0000001):
    return stats.norm.cdf(muon_energy, loc=minimum_muon_energy(ice_distance)*Units.GeV,
                          scale=scale*minimum_muon_energy(ice_distance)*Units.GeV)
    # return muon_energy > minimum_muon_energy(ice_distance)*Units.GeV
