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

