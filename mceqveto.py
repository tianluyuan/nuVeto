#!/usr/bin/env python

"""
Calculate the rate at which atmospheric neutrinos arrive at an underground
detector with no accompanying muons.
"""
from collections import namedtuple
from enum import Enum
from functools32 import lru_cache
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
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


# Global Barr parameter table 
# format [(x_min, x_max, E_min, E_max)] | x is x_lab= E_pi/E, E projectile-air interaction energy
ParamInfo = namedtuple('ParamInfo', 'regions error pdg')
BARR = {
    'a': ParamInfo([(0.0, 0.5, 0.00, 8.0)], 0.1, 211),
    'b1': ParamInfo([(0.5, 1.0, 0.00, 8.0)], 0.3, 211),
    'b2': ParamInfo([(0.6, 1.0, 8.00, 15.0)], 0.3, 211),
    'c': ParamInfo([(0.2, 0.6, 8.00, 15.0)], 0.1, 211),
    'd1': ParamInfo([(0.0, 0.2, 8.00, 15.0)], 0.3, 211),
    'd2': ParamInfo([(0.0, 0.1, 15.0, 30.0)], 0.3, 211),
    'd3': ParamInfo([(0.1, 0.2, 15.0, 30.0)], 0.1, 211),
    'e': ParamInfo([(0.2, 0.6, 15.0, 30.0)], 0.05, 211),
    'f': ParamInfo([(0.6, 1.0, 15.0, 30.0)], 0.1, 211),
    'g': ParamInfo([(0.0, 0.1, 30.0, 1e11)], 0.3, 211),
    'h1': ParamInfo([(0.1, 1.0, 30.0, 500.)], 0.15, 211),
    'h2': ParamInfo([(0.1, 1.0, 500.0, 1e11)], 0.15, 211),
    'i': ParamInfo([(0.1, 1.0, 500.0, 1e11)], 0.122, 211),
    'w1': ParamInfo([(0.0, 1.0, 0.00, 8.0)], 0.4, 321),
    'w2': ParamInfo([(0.0, 1.0, 8.00, 15.0)], 0.4, 321),
    'w3': ParamInfo([(0.0, 0.1, 15.0, 30.0)], 0.3, 321),
    'w4': ParamInfo([(0.1, 0.2, 15.0, 30.0)], 0.2, 321),
    'w5': ParamInfo([(0.0, 0.1, 30.0, 500.)], 0.4, 321),
    'w6': ParamInfo([(0.0, 0.1, 500., 1e11)], 0.4, 321),
    'x': ParamInfo([(0.2, 1.0, 15.0, 30.0)], 0.1, 321),
    'y1': ParamInfo([(0.1, 1.0, 30.0, 500.)], 0.3, 321),
    'y2': ParamInfo([(0.1, 1.0, 500., 1e11)], 0.3, 321),
    'z': ParamInfo([(0.1, 1.0, 500., 1e11)], 0.122, 321),
    'ch_a': ParamInfo([(0.0, 0.1, 0., 1e11)], 0.1, 411), # these uncertainties are from A. Fedynitch Vietnus
    'ch_b': ParamInfo([(0.1, 1.0, 0., 1e11)], 0.7, 411),
    'ch_e': ParamInfo([(0.1, 1.0, 800., 1e11)], 0.25, 411)
}


def barr_unc(xmat, egrid, pname, value):
    """Implementation of hadronic uncertainties as in Barr et al. PRD 74 094009 (2006)

    The names of parameters are explained in Fig. 2 and Fig. 3 in the paper."""


    # Energy dependence
    u = lambda E, val, ethr, maxerr, expected_err: val*min(
        maxerr/expected_err,
        0.122/expected_err*np.log10(E / ethr)) if E > ethr else 0.

    modmat = np.ones_like(xmat)
    modmat[np.tril_indices(xmat.shape[0], -1)] = 0.

    for minx, maxx, mine, maxe in BARR[pname].regions:
        eidcs = np.where((mine < egrid) & (egrid <= maxe))[0]
        for eidx in eidcs:
            xsel = np.where((xmat[:eidx + 1, eidx] >= minx) &
                            (xmat[:eidx + 1, eidx] <= maxx))[0]
            if not np.any(xsel):
                continue
            if pname in ['i', 'z']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 500., 0.5, 0.122)
            elif pname in ['ch_e']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 800., 0.3, 0.25)
            else:
                modmat[xsel, eidx] += value

    return modmat


def amu(particle):
    """
    :param particle: primary particle's corsika id

    :returns: the atomic mass of particle
    """
    return 1 if particle==14 else particle/100


def ice(cos_theta, depth=1950, elevation=2400):
    """ Returns the in-ice distance for a detector at depth.

    :param cos_theta: cosine of zenith angle (in detector-centered coordinates)
    :param depth:     depth of detector (in meters below the surface)
    :param elevation: elevation of the ice surface above sea level (meters)
    """
    r = 6356752. + elevation
    x = r-depth
    cos2th = 2*cos_theta**2 - 1
    a = r**2+x**2*cos2th

    return np.sqrt(a-np.sqrt(2)*np.sqrt(x**2*cos_theta**2*(2*depth*r-depth**2+a)))


def minimum_muon_energy(distance):
    """
    Minimum muon energy required to survive the given thickness of ice with at
    least 1 TeV 50% of the time.

    :returns: minimum muon energy [GeV]
    """
    # require that the muon have median energy 1 TeV
    b, c = 2.52151, 7.13834
    return 1e3 * np.exp(1e-3 * distance / (b) + 1e-8 * (distance**2) / c)


def mcsolver(primary_energy, cos_theta, particle, pmods=(), hadr='SIBYLL2.3c'):
    """wrapper fn to protect against modifying hadron production rates
    (Barr et. al) for non-proton primaries. Only proton yields are
    affected. This is to speed up the caching of outputs from
    mcsolver_wrapped.
    """
    if particle == 14:
        mods = pmods
    else:
        mods = ()
    return mcsolver_wrapped(primary_energy, cos_theta, particle, mods, hadr)


@lru_cache(maxsize=1024)
def mcsolver_wrapped(primary_energy, cos_theta, particle, pmods=(), hadr='SIBYLL2.3c'):
    Info = namedtuple('Info', 'e_grid e_widths')
    Yields = namedtuple('Yields', 'mu numu nue conv_numu conv_nue pr_numu pr_nue')
    MCEQ.set_single_primary_particle(primary_energy, particle)
    MCEQ.set_theta_deg(np.degrees(np.arccos(cos_theta)))
    MCEQ.set_interaction_model(hadr)

    # In case there was something before, reset modifications
    MCEQ.unset_mod_pprod(dont_fill=True)
    for pmod in pmods:
        # Modify proton-air -> mod[0]
        MCEQ.set_mod_pprod(2212,BARR[pmod[0]].pdg,barr_unc,pmod,delay_init=True)
    # Populate the modifications to the matrices by re-filling the interaction matrix
    MCEQ._init_default_matrices(skip_D_matrix=True)
    # Print the changes
    # print "\n \n This is the printout from the print_mod_pprod routine"
    # MCEQ.y.print_mod_pprod()
    # print "\n \n"
    MCEQ.solve()

    # en = primary_energy/amu(particle)
    # x = MCEQ.e_grid/en

    mu = MCEQ.get_solution('total_mu-') + MCEQ.get_solution('total_mu+')
    numu = MCEQ.get_solution('total_numu')+MCEQ.get_solution('total_antinumu')
    nue = MCEQ.get_solution('total_nue')+MCEQ.get_solution('total_antinue')
    conv_numu = MCEQ.get_solution('conv_numu')+MCEQ.get_solution('conv_antinumu')
    conv_nue = MCEQ.get_solution('conv_nue')+MCEQ.get_solution('conv_antinue')
    pr_numu = MCEQ.get_solution('pr_numu')+MCEQ.get_solution('pr_antinumu')
    pr_nue = MCEQ.get_solution('pr_nue')+MCEQ.get_solution('pr_antinue')
    return Info(MCEQ.e_grid, MCEQ.e_widths), Yields(mu, numu, nue, conv_numu, conv_nue, pr_numu, pr_nue)


def mceq_yield(primary_energy, cos_theta, particle, kind='mu', pmods=(), hadr='SIBYLL2.3c'):
    Solution = namedtuple('Solution', 'info yields')
    info, mcs = mcsolver(primary_energy, cos_theta, particle, pmods, hadr)
    return Solution(info, mcs._asdict()[kind])


def flux(primary_energy, particle):
    """ Primary flux
    """
    pmod = SETUP['flux'](SETUP['gen'])
    return pmod.nucleus_flux(particle, primary_energy)


def response_function(primary_energy, cos_theta, particle, elep, kind='mu', pmods=(), hadr='SIBYLL2.3c'):
    """ response function in https://arxiv.org/pdf/1405.0525.pdf
    """
    sol = mceq_yield(primary_energy, cos_theta, particle, kind, pmods, hadr)
    fnsol = interp1d(sol.info.e_grid, sol.yields, kind='quadratic',
                     assume_sorted=True) 
    return flux(primary_energy, particle)*fnsol(elep)


def prob_nomu(primary_energy, cos_theta, particle, pmods=(), hadr='SIBYLL2.3c'):
    emu_min = minimum_muon_energy(ice(cos_theta))
    mu = mceq_yield(primary_energy, cos_theta, particle, 'mu', pmods, hadr)
    above = mu.info.e_grid > emu_min
    return np.exp(-simps(mu.yields[above], mu.info.e_grid[above]))


def passing_rate(enu, cos_theta, kind='numu', pmods=(), hadr='SIBYLL2.3c', accuracy=20):
    pmod = SETUP['flux'](SETUP['gen'])
    passed = 0
    total = 0
    for particle in pmod.nucleus_ids:
        # A continuous input energy range is allowed between
        # :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.
        eprimaries = amu(particle)*np.logspace(2, 10, accuracy) 
        numer = []
        denom = []
        for primary_energy in eprimaries:
            res = response_function(primary_energy, cos_theta, particle, enu, kind, pmods, hadr)
            pnm = prob_nomu(primary_energy, cos_theta, particle, pmods, hadr)
            numer.append(res*pnm)
            denom.append(res)

        passed += np.trapz(numer, eprimaries)
        total += np.trapz(denom, eprimaries)
    return passed/total
