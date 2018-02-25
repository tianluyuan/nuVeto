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
from barr_uncertainties import *
from utils import *


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
GEOM = Geometry(1950*Units.m)
MU = MuonProb('external/mmc/prpl.pkl')


# def mcsolver(primary_energy, cos_theta, particle, pmods=(), hadr='SIBYLL2.3c'):
#     """wrapper fn to protect against modifying hadron production rates
#     (Barr et. al) for non-proton primaries. Only proton yields are
#     affected. This is to speed up the caching of outputs from
#     mcsolver_wrapped.
#     """
#     if particle == 14:
#         mods = pmods
#     else:
#         mods = ()
#     return mcsolver_wrapped(primary_energy, cos_theta, particle, mods, hadr)


@lru_cache(maxsize=2**12)
def mcsolver(primary_energy, cos_theta, particle, pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c'):
    Info = namedtuple('Info', 'e_grid e_widths')
    Yields = namedtuple('Yields', 'mu numu nue conv_numu conv_nue pr_numu pr_nue')
    MCEQ.set_primary_model(*pmodel)
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


def mceq_yield(primary_energy, cos_theta, particle, kind='mu', pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c'):
    Solution = namedtuple('Solution', 'info yields')
    info, mcs = mcsolver(primary_energy, cos_theta, particle, pmods, pmodel, hadr)
    return Solution(info, mcs._asdict()[kind])


def flux(primary_energy, particle, pmodel):
    """ Primary flux
    """
    pmod = pmodel[0](pmodel[1])
    return pmod.nucleus_flux(particle, primary_energy)


def response_function(primary_energy, cos_theta, particle, elep, kind='mu', pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c'):
    """ response function in https://arxiv.org/pdf/1405.0525.pdf
    """
    sol = mceq_yield(primary_energy, cos_theta, particle, kind, pmods, pmodel, hadr)
    fnsol = interp1d(sol.info.e_grid, sol.yields, kind='quadratic',
                     assume_sorted=True)
    return flux(primary_energy, particle, pmodel)*fnsol(elep)


def prob_nomu_simple(primary_energy, cos_theta, particle, enu, pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', nenu=2):
    # only subtract if it matters
    if nenu*enu > 0.01*primary_energy:
        primary_energy -= nenu*enu
    emu_min = minimum_muon_energy(GEOM.overburden(cos_theta))
    if primary_energy < emu_min:
        return 1
    mu = mceq_yield(primary_energy, cos_theta, particle, 'mu', pmods, pmodel, hadr)
    if mu.info.e_grid[-1] < emu_min:
        # probability of no muons that make it will be 1 if emu_min > highest yield
        return 1
    fnmu = interp1d(mu.info.e_grid, mu.yields, kind='quadratic',
                    assume_sorted=True)
    above = mu.info.e_grid > emu_min

    # idx = max(0,np.argmax(mu.info.e_grid > emu_min)-1)
    # return np.exp(-simps(mu.yields[idx:], mu.info.e_grid[idx:]))
    return np.exp(-np.trapz(np.concatenate(([fnmu(emu_min)],mu.yields[above])),
                            np.concatenate(([emu_min],mu.info.e_grid[above]))))


def prob_nomu(primary_energy, cos_theta, particle, enu, pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', nenu=2):
    # only subtract if it matters
    if nenu*enu > 0.01*primary_energy:
        primary_energy -= nenu*enu

    l_ice = GEOM.overburden(cos_theta)
    mu = mceq_yield(primary_energy, cos_theta, particle, 'mu', pmods, pmodel, hadr)

    coords = zip(mu.info.e_grid*Units.GeV, [l_ice]*len(mu.info.e_grid))
    return np.exp(-np.trapz(mu.yields*MU.prpl(coords),
                            mu.info.e_grid))


def passing_rate(enu, cos_theta, kind='numu', pmods=(), pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=20, fraction=True, nenu=2, prpl=False):
    pmod = pmodel[0](pmodel[1])
    passed = 0
    total = 0
    for particle in pmod.nucleus_ids:
        # A continuous input energy range is allowed between
        # :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.
        eprimaries = amu(particle)*np.logspace(2, 10, accuracy)
        numer = []
        denom = []
        istart = max(0, np.argmax(eprimaries > enu) - 1)
        for primary_energy in eprimaries[istart:]:
            res = response_function(primary_energy, cos_theta, particle, enu, kind, pmods, pmodel, hadr)
            if prpl:
                pnm = prob_nomu(primary_energy, cos_theta, particle, enu, pmods, pmodel, hadr, nenu)
            else:
                pnm = prob_nomu_simple(primary_energy, cos_theta, particle, enu, pmods, pmodel, hadr, nenu)
            numer.append(res*pnm)
            denom.append(res)

        passed += np.trapz(numer, eprimaries[istart:])
        total += np.trapz(denom, eprimaries[istart:])
    return passed/total if fraction else passed
