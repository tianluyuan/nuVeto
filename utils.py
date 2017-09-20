import numpy as np


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
