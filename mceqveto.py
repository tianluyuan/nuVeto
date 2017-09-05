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
from functools32 import lru_cache
import numpy as np
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without


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
    Solutions = namedtuple('Solutions', 'x mu numu nue charm')
    mceq_run = MCEqRun(
        # provide the string of the interaction model
    interaction_model='SIBYLL2.3',
        # primary cosmic ray flux model
        # support a tuple (primary model class (not instance!), arguments)
    primary_model=(pm.HillasGaisser2012, "H3a"),
        # Zenith angle in degrees. 0=vertical, 90=horizontal
    theta_deg=np.degrees(np.arccos(cos_theta)),
        # expand the rest of the options from mceq_config.py
    **config)

    mceq_run.set_single_primary_particle(primary_energy, particle)
    mceq_run.solve()

    x = mceq_run.e_grid/primary_energy

    mu = mceq_run.get_solution('mu-', 0) + mceq_run.get_solution('mu+',0)
    numu = mceq_run.get_solution('conv_numu', 0)+mceq_run.get_solution('conv_antinumu',0)
    nue = mceq_run.get_solution('conv_nue', 0)+mceq_run.get_solution('conv_antinue',0)
    charm = mceq_run.get_solution('pr_numu', 0)+mceq_run.get_solution('pr_antinumu',0) \
        + mceq_run.get_solution('pr_nue', 0)+mceq_run.get_solution('pr_antinue',0) 
    return Solutions(x, mu, numu, nue, charm)


def mceq_yield(primary_energy, cos_theta, particle, kind='mu'):
    mcs = mcsolver(primary_energy, cos_theta, particle)
    if kind == 'mu':
        return mcs.x, mcs.mu
    elif kind == 'numu':
        return mcs.x, mcs.numu
    elif kind == 'nue':
        return mcs.x, mcs.nue
    elif kind == 'charm':
        return mcs.x, mcs.charm


class ParticleType(object):
    PPlus = 14
    He4Nucleus = 402
    N14Nucleus = 1407
    Al27Nucleus = 2713
    Fe56Nucleus = 5626
