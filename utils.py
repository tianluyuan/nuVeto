import numpy as np


class Units(object):
    # units
    Na = 6.0221415e+23 # mol/cm^3
    km = 5.0677309374099995 # km to GeV^-1 value from SQuIDS
    cm = km*1.e-5
    m = km*1.e-3
    gr = 5.62e23 # gr to GeV value from SQuIDS
    sec = 1523000.0 #$ sec to GeV^-1 from SQuIDS
    GeV = 1
    TeV = 1.e3*GeV
    PeV = 1.e3*TeV


class ParticleProperties(object):
    r_dict ={}; mass_dict = {}; lifetime_dict = {}; pdg_id = {};
    r_dict["kaon"]=0.046
    r_dict["pion"]=0.573

    mass_dict["kaon"]=0.493677*Units.GeV # GeV
    mass_dict["pion"]=0.139570*Units.GeV # GeV
    mass_dict["D"]=1.86962*Units.GeV # GeV
    mass_dict["air"]=(14.5)*Units.GeV # GeV

    lifetime_dict["kaon"]=1.2389e-8*Units.sec # s
    lifetime_dict["pion"]=2.6033e-8*Units.sec # 
    lifetime_dict["D"]=1.040e-12*Units.sec # seconds to usual Units

    pdg_id["D"] = 411 # D+
    pdg_id["kaon"] = 321 # k+
    pdg_id["pion"] = 211 # pi+
    pdg_id["numu"] = 14
    pdg_id["nue"] = 12


class MaterialProperties(object):
    a = {}; b = {}; density = {};
    a["ice"]=0.249*Units.GeV/Units.m # GeV/mwe
    a["rock"]=0.221*Units.GeV/Units.m # GeV/mwe
    b["ice"]=0.422e-3/Units.m # 1/mwe
    b["rock"]=0.531e-3/Units.m # 1/mwe
    density["ice"] = 0.9167*Units.gr/Units.cm**3 # g/cm^3


def amu(particle):
    """
    :param particle: primary particle's corsika id

    :returns: the atomic mass of particle
    """
    return 1 if particle==14 else particle/100


def centers(x):
    return (x[:-1]+x[1:])*0.5


def overburden(cos_theta, depth=1950, elevation=2400):
    """Returns the overburden for a detector at *depth* below some surface
    at *elevation*.

    From law of cosines,
    x^2 == r^2+(r-d)^2-2r(r-d)cos(gamma)
    where
    r*cos(gamma) = r-d+x*cos(theta), solve and return x.

    :param cos_theta: cosine of zenith angle 
    :param depth:     depth of detector (in meters below the surface)
    :param elevation: elevation of the surface above sea level (meters)
    """
    d = depth
    r = 6356752. + elevation
    z = r-d

    return np.sqrt(z**2*cos_theta**2+d*(2*r-d))-z*cos_theta


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


def muon_reach_prob(muon_energy, ice_distance):
    if(muon_energy > minimum_muon_energy(ice_distance)*Units.GeV):
        return 1.
    else:
        return 0.
    # simplifying assumption that the muon reach distribution is a gaussian
    # the probability that it does not reach is given by the cumnulative distribution function
    # on the other hand the reaching probability is given by the survival distribution function
    # the former is associated with the passing rate.
    return stats.norm.cdf(ice_column_density/MaterialProperties.density["ice"],
            loc=MeanMuonDistance(muon_energy),scale=np.sqrt(MeanMuonDistance(muon_energy)))
