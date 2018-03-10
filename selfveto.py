from functools32 import lru_cache
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without
from utils import *


SETUP = {'pmodel':(pm.HillasGaisser2012,'H3a'),
         'hadr':'SIBYLL2.3c'}
ADV_SET = config['adv_set']
ADV_SET['no_mixing'] = False
MCEQ = MCEqRun(
    # provide the string of the interaction model
    interaction_model=SETUP['hadr'],
    # primary cosmic ray flux model
    # support a tuple (primary model class (not instance!), arguments)
    primary_model=SETUP['pmodel'],
    # zenith angle \theta in degrees, measured positively from vertical direction
    theta_deg = 0.,
    adv_set = ADV_SET,
    compact_mode = False,
    # expand the rest of the options from mceq_config.py
    **mceq_config_without(['compact_mode', 'adv_set']))
GEOM = Geometry(1950*Units.m)

@lru_cache(maxsize=2**12)
def get_dNdEE(mother, daughter):
    ihijo = 20
    e_grid = MCEQ.e_grid
    delta = MCEQ.e_widths
    x_range = e_grid[ihijo]/e_grid
    rr = ParticleProperties.rr(mother, daughter)
    dN_mat = MCEQ.decays.get_d_matrix(ParticleProperties.pdg_id[mother],
                                      ParticleProperties.pdg_id[daughter])
    dNdEE = dN_mat[ihijo]*e_grid/delta
    logx = np.log10(x_range)
    logx_width = -np.diff(logx)[0]
    good = (logx + logx_width/2 < np.log10(1-rr)) & (x_range >= 1.e-3)
    if (mother == 'pi+' and daughter == 'numu') or (mother == 'pi-' and daughter == 'antinumu'):
        # pi -> numu are all 2-body
        dNdEE_edge = 1/(1-rr)
    elif (mother == 'K+' and daughter == 'numu') or (mother == 'K-' and daughter == 'antinumu'):
        # K -> numu are mostly 2-body
        dNdEE_edge = 0.6356/(1-rr)
    else:
        # everything else 3-body
        dNdEE_edge = 0.
        
    lower = dNdEE[(x_range < 1-rr) & (x_range >= 1.0e-3)][-1]
    dNdEE_interp = interpolate.interp1d(
        np.concatenate([[1-rr], x_range[good]]),
        np.concatenate([[dNdEE_edge], dNdEE[good]]), kind='quadratic',
        bounds_error=False, fill_value=(lower, 0.0))
    return x_range, dNdEE, dNdEE_interp


@lru_cache(maxsize=2**12)
def solver(cos_theta, pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c'):
    """This will cache MCEQ solutions for each combination of the
    arguments. Ensure that it returns everything that changes with different arguments
    """
    theta = np.degrees(np.arccos(GEOM.cos_theta_eff(cos_theta)))
    MCEQ.set_primary_model(*pmodel)
    MCEQ.set_interaction_model(hadr)
    MCEQ.set_theta_deg(theta)

    x_vec = np.logspace(np.log10(1), np.log10(MCEQ.density_model.max_X), 11)
    heights = MCEQ.density_model.X2h(x_vec)
    lengths = MCEQ.density_model.geom.delta_l(heights, np.radians(theta)) * Units.cm
    deltahs = np.diff(lengths)
    MCEQ.solve(int_grid=x_vec[:-1], grid_var="X")
    return deltahs, x_vec[:-1], MCEQ.grid_sol


def get_solution_orig(grid_sol,
                      particle_name,
                      xv,
                      mag=0.,
                      grid_idx=None,
                      integrate=False):
    """Retrieves solution of the calculation on the energy grid.

    Some special prefixes are accepted for lepton names:

    - the total flux of muons, muon neutrinos etc. from all sources/mothers
      can be retrieved by the prefix ``total_``, i.e. ``total_numu``
    - the conventional flux of muons, muon neutrinos etc. from all sources
      can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
    - correspondigly, the flux of leptons which originated from the decay
      of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``
    - conventional leptons originating neither from pion nor from kaon
      decay are collected in a category without any prefix, e.g. ``numu`` or
      ``mu+``

    Args:
      particle_name (str): The name of the particle such, e.g.
        ``total_mu+`` for the total flux spectrum of positive muons or
        ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
      mag (float, optional): 'magnification factor': the solution is
        multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
      grid_idx (int, optional): if the integrator has been configured to save
        intermediate solutions on a depth grid, then ``grid_idx`` specifies
        the index of the depth grid for which the solution is retrieved. If
        not specified the flux at the surface is returned
      integrate (bool, optional): return averge particle number instead of
      flux (multiply by bin width)

    Returns:
      (numpy.array): flux of particles on energy grid :attr:`e_grid`
    """
    res = np.zeros(MCEQ.d)
    ref = MCEQ.pname2pref
    sol = None
    if grid_idx is None:
        sol = grid_sol[-1]
    elif grid_idx >= len(grid_sol):
        sol = grid_sol[-1]
    else:
        sol = grid_sol[grid_idx]

    if particle_name.startswith('total'):
        lep_str = particle_name.split('_')[1]
        for prefix in ('pr_', 'pi_', 'k_', ''):
            particle_name = prefix + lep_str
            res += sol[ref[particle_name].lidx():
                       ref[particle_name].uidx()] * \
                MCEQ.e_grid ** mag
    elif particle_name.startswith('conv'):
        lep_str = particle_name.split('_')[1]
        for prefix in ('pi_', 'k_', ''):
            particle_name = prefix + lep_str
            res += sol[ref[particle_name].lidx():
                       ref[particle_name].uidx()] * \
                MCEQ.e_grid ** mag
    else:
        res = sol[ref[particle_name].lidx():
                  ref[particle_name].uidx()] * \
            MCEQ.e_grid ** mag

    if not integrate:
        return res
    else:
        return res * MCEQ.e_widths

    
def get_solution(grid_sol,
                 particle_name,
                 xv,
                 mag=0.,
                 grid_idx=None,
                 integrate=False):
    """Retrieves solution of the calculation on the energy grid.

    Args:
      particle_name (str): The name of the particle such, e.g.
        ``total_mu+`` for the total flux spectrum of positive muons or
        ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
      mag (float, optional): 'magnification factor': the solution is
        multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
      grid_idx (int, optional): if the integrator has been configured to save
        intermediate solutions on a depth grid, then ``grid_idx`` specifies
        the index of the depth grid for which the solution is retrieved. If
        not specified the flux at the surface is returned
      integrate (bool, optional): return averge particle number instead of
      flux (multiply by bin width)

    Returns:
      (numpy.array): flux of particles on energy grid :attr:`e_grid`
    """
    res = np.zeros(MCEQ.d)
    ref = MCEQ.pname2pref
    sol = None
    if grid_idx is None:
        sol = grid_sol[-1]
    elif grid_idx >= len(grid_sol):
        sol = grid_sol[-1]
    else:
        sol = grid_sol[grid_idx]

    res = np.array([0.]*len(MCEQ.e_grid))
    part_xs = MCEQ.cs.get_cs(ParticleProperties.pdg_id[particle_name])
    rho_air = MCEQ.density_model.X2rho(xv)
    # meson decay length
    decayl = (MCEQ.e_grid * Units.GeV)/ParticleProperties.mass_dict[particle_name] * ParticleProperties.lifetime_dict[particle_name] /Units.cm
    # meson interaction length
    interactionl = 1/(MCEQ.cs.get_cs(ParticleProperties.pdg_id[particle_name])*rho_air*Units.Na/Units.mol_air)
    # number of targets per cm2
    ndens = rho_air*Units.Na/Units.mol_air
    for prim in ['p', 'p-bar', 'n', 'n-bar']:
        prim_flux = sol[ref[prim].lidx():
                        ref[prim].uidx()]
        prim_xs = MCEQ.cs.get_cs(ParticleProperties.pdg_id[prim])
        int_yields = MCEQ.y.get_y_matrix(
            ParticleProperties.pdg_id[prim],
            ParticleProperties.pdg_id[particle_name])
        res += np.dot(int_yields,
                      prim_flux*prim_xs*ndens)
    res *= decayl
    # combine with direct
    direct = sol[ref[particle_name].lidx():
                 ref[particle_name].uidx()]
    res = np.concatenate((res[direct<=0], direct[direct>0]))

    res *= MCEQ.e_grid ** mag

    if not integrate:
        return res
    else:
        return res * MCEQ.e_widths


def categ_to_mothers(categ, daughter):
    charge = '-' if 'anti' in daughter else '+'
    bar = '-bar' if 'anti' in daughter else ''
    lbar = '-bar' if 'anti' not in daughter else ''
    if categ == 'conv':
        mothers = ['pi'+charge, 'K'+charge, 'K0L'] #K0S in uncorrelated?
        if 'nue' in daughter:
            mothers.append('K0S')
    elif categ == 'pr':
        mothers = ['D'+charge, 'Ds'+charge, 'D0'+bar, 'Lambda0'+lbar]
    else:
        mothers = [categ,]
    return mothers
    

def passing_rate(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, prpl='step_1'):
    def get_rescale_phi(mother, dh, xv, grid_sol, idx):
        inv_decay_length_array = (ParticleProperties.mass_dict[mother] / (MCEQ.e_grid * Units.GeV)) *(dh / ParticleProperties.lifetime_dict[mother])
        rescale_phi = inv_decay_length_array * get_solution(grid_sol, mother, xv, grid_idx=idx)
        return interpolate.interp1d(MCEQ.e_grid, rescale_phi, kind='quadratic', fill_value='extrapolate')

    def get_integrand(categ, daughter, dh, xv, grid_sol, idx, weight_fn, esamp):
        mothers = categ_to_mothers(categ, daughter)
        ys = np.zeros(len(esamp))
        for mother in mothers:
            dNdEE = get_dNdEE(mother, daughter)[-1]
            rescale_phi = get_rescale_phi(mother, dh, xv, grid_sol, idx)
            ys += dNdEE(enu/esamp)/esamp*rescale_phi(esamp)*weight_fn(esamp)
            
        return ys

    categ, daughter = kind.split('_')
    
    ice_distance = GEOM.overburden(cos_theta)
    identity = lambda Ep: 1
    if 'numu' not in daughter:
        # muon accompanies numu only
        reaching = identity
    else:
        fn = MuonProb(prpl)
        reaching = lambda Ep: 1. - fn.prpl(zip((Ep-enu)*Units.GeV,
                                               [ice_distance]*len(Ep)))

    deltahs, x_vec, grid_sol = solver(cos_theta, pmodel, hadr)
    passing_numerator = 0
    passing_denominator = 0
    esamp = np.logspace(np.log10(enu), np.log10(MCEQ.e_grid[-1]), int(10**accuracy))
    for idx, (dh, xv) in enumerate(zip(deltahs, x_vec)):
        passing_numerator += integrate.trapz(
            get_integrand(categ, daughter, dh, xv, grid_sol, idx, reaching, esamp), esamp)
        passing_denominator += integrate.trapz(
            get_integrand(categ, daughter, dh, xv, grid_sol, idx, identity, esamp), esamp)
        # print passing_numerator, passing_denominator
    return passing_numerator/passing_denominator if fraction else passing_denominator
