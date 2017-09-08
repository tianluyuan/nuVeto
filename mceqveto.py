#!/usr/bin/env python

"""
Calculate the rate at which atmospheric neutrinos arrive at an underground
detector with no accompanying muons.
"""

# Copyright (c) 2014, Jakob van Santen <jvansanten@icecube.wisc.edu>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from collections import namedtuple
from enum import Enum
from functools32 import lru_cache
import numpy as np
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without


SETUP = {'flux':pm.HillasGaisser2012,
         'gen':'H3a',
         'hadr':'SIBYLL2.3c'}
MCEQ = MCEqRun(
    # provide the string of the interaction model
    interaction_model=SETUP['hadr'],
    # primary cosmic ray flux model
    # support a tuple (primary model class (not instance!), arguments)
    primary_model=(SETUP['flux'], SETUP['gen']),
    # zenith angle \theta in degrees, measured positively from vertical direction
    theta_deg = 0.,
    # expand the rest of the options from mceq_config.py
    **config)


Kinds = Enum('Kinds', 'mu numu nue charm')


def amu(particle):
    """
    :param particle: primary particle's corsika id

    :returns: the atomic mass of particle
    """
    return 1 if particle==14 else particle/100

    
def overburden(cos_theta, depth=1950, elevation=2400):
    """
    Overburden for a detector buried underneath a flat surface.

    :param cos_theta: cosine of zenith angle (in detector-centered coordinates)
    :param depth:     depth of detector (in meters below the surface)
    :param elevation: elevation of the surface above sea level

    :returns: an overburden [meters]
    """
    # curvature radius of the surface (meters)
    r = 6371315 + elevation
    # this is secrety a translation in polar coordinates
    return (np.sqrt(2 * r * depth + (cos_theta * (r - depth))**2 - depth**2) - (r - depth) * cos_theta)


def minimum_muon_energy(distance):
    """
    Minimum muon energy required to survive the given thickness of ice with at
    least 1 TeV 50% of the time.

    :returns: minimum muon energy [GeV]
    """
    # require that the muon have median energy 1 TeV
    b, c = 2.52151, 7.13834
    return 1e3 * np.exp(1e-3 * distance / (b) + 1e-8 * (distance**2) / c)


def effective_costheta(costheta):
    """
    Effective local atmospheric density correction from [Chirkin]_.

    .. [Chirkin] D. Chirkin. Fluxes of atmospheric leptons at 600-GeV - 60-TeV. 2004. http://arxiv.org/abs/hep-ph/0407078
    """
    x = costheta
    p = [0.102573, -0.068287, 0.958633, 0.0407253, 0.817285]
    return np.sqrt((x**2 + p[0]**2 + p[1] * x**p[2] + p[3] * x**p[4]) / (1 + p[0]**2 + p[1] + p[3]))


class fpe_context(object):
    """
    Temporarily modify floating-point exception handling
    """

    def __init__(self, **kwargs):
        self.new_kwargs = kwargs

    def __enter__(self):
        self.old_kwargs = np.seterr(**self.new_kwargs)

    def __exit__(self, *args):
        np.seterr(**self.old_kwargs)


@lru_cache(maxsize=512)
def mcsolver(primary_energy, cos_theta, particle):
    Info = namedtuple('Info', 'e_grid e_widths')
    Yields = namedtuple('Yields', 'mu numu nue charm')
    MCEQ.set_single_primary_particle(primary_energy, particle)
    MCEQ.set_theta_deg(np.degrees(np.arccos(cos_theta)))
    MCEQ.solve()

    # en = primary_energy/amu(particle)
    # x = MCEQ.e_grid/en

    mu = MCEQ.get_solution('total_mu-', 0) + MCEQ.get_solution('total_mu+',0)
    numu = MCEQ.get_solution('conv_numu', 0)+MCEQ.get_solution('conv_antinumu',0)
    nue = MCEQ.get_solution('conv_nue', 0)+MCEQ.get_solution('conv_antinue',0)
    charm = MCEQ.get_solution('pr_numu', 0)+MCEQ.get_solution('pr_antinumu',0) \
        + MCEQ.get_solution('pr_nue', 0)+MCEQ.get_solution('pr_antinue',0) 
    return Info(MCEQ.e_grid, MCEQ.e_widths), Yields(mu, numu, nue, charm)


def mceq_yield(primary_energy, cos_theta, particle, kind=Kinds.mu):
    Solution = namedtuple('Solution', 'info yields')
    info, mcs = mcsolver(primary_energy, cos_theta, particle)
    if kind == Kinds.mu:
        return Solution(info, mcs.mu)
    elif kind == Kinds.numu:
        return Solution(info, mcs.numu)
    elif kind == Kinds.nue:
        return Solution(info, mcs.nue)
    elif kind == Kinds.charm:
        return Solution(info, mcs.charm)


def flux(primary_energy, particle):
    """ Primary flux
    """
    pmod = SETUP['flux'](SETUP['gen'])
    return pmod.nucleus_flux(particle, primary_energy)


def response_function(primary_energy, cos_theta, particle, elep, kind=Kinds.mu):
    """ response function in https://arxiv.org/pdf/1405.0525.pdf
    """
    sol = mceq_yield(primary_energy, cos_theta, particle, kind=kind)
    return flux(primary_energy, particle)*np.interp(elep, sol.info.e_grid, sol.yields)


def prob_nomu(primary_energy, cos_theta, particle):
    emu_min = minimum_muon_energy(overburden(cos_theta))
    mu = mceq_yield(primary_energy, cos_theta, particle, kind=Kinds.mu)
    above = mu.info.e_grid > emu_min
    return np.exp(-np.trapz(mu.yields[above], mu.info.e_grid[above]))


def passing_rate(enu, cos_theta, kind=Kinds.numu):
    pmod = SETUP['flux'](SETUP['gen'])

    epedges = np.logspace(2, 11, 10)
    epcenters = 10**((np.log10(epedges[:-1])+np.log10(epedges[1:]))/2)
    
    passed = 0
    total = 0
    for particle in pmod.nucleus_ids:
        numer = []
        denom = []
        for primary_energy in epcenters:
            res = response_function(primary_energy, cos_theta, particle, enu, kind=kind)
            pnm = prob_nomu(primary_energy, cos_theta, particle)
            numer.append(res*pnm)
            denom.append(res)

        passed += np.trapz(numer, epcenters)
        total += np.trapz(denom, epcenters)
    return passed/total
