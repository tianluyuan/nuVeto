from functools32 import lru_cache
import numpy as np
import tqdm
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without
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


@lru_cache(maxsize=2**12)
def get_dNdEE(mother, daughter):
    ihijo = 20
    e_grid = MCEQ.e_grid
    delta = MCEQ.e_widths
    x_range = e_grid[ihijo]/e_grid
    rr = ParticleProperties.rr(mother, daughter)
    dN_mat = MCEQ.ds.get_d_matrix(ParticleProperties.pdg_id[mother],
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
        np.concatenate([[dNdEE_edge], dNdEE[good]]),
        bounds_error=False, fill_value=(lower, 0.0))
    return x_range, dNdEE, dNdEE_interp


def get_dN(integrand):
    e_bins = MCEQ.y.e_bins
    e_grid = MCEQ.e_grid
    RY_matrix = np.zeros((len(e_grid), len(e_grid)))
    for ipadre in tqdm.tqdm(range(len(e_grid))):
        for ihijo in range(len(e_grid)):
            # print "doing " + str(ipadre) + " " + str(ihijo)
            if ihijo > ipadre:
                RY_matrix[ihijo][ipadre] = 0
                continue

            EnuMin = e_bins[ihijo]
            EnuMax = e_bins[ihijo+1]
            EpMin = e_bins[ipadre]
            EpMax = e_bins[ipadre+1]

            # integrand = lambda Ep, Enu: (dNdEE_Interpolator(
            #     Enu / Ep) / Ep) * (1. - muon_reach_prob((Ep - Enu) * Units.GeV, ice_distance))
            # xen = np.linspace(EnuMin, EnuMax, 100*np.log10(EnuMax))
            # xep = np.linspace(EpMin, EpMax, 100*np.log10(EpMax))
            # dep = xep[1]-xep[0]
            # for ep in xep:
            #     RY_matrix[ihijo][ipadre]+=integrate.trapz(
            #         integrand(ep, xen), xen)/(EnuMax-EnuMin)*dep
            RY_matrix[ihijo][ipadre] = integrate.dblquad(
                integrand,
                EnuMin, EnuMax,
                lambda Enu: np.max([Enu, EpMin]),
                lambda Enu: EpMax,
                epsabs=0, epsrel=1.e-2)[0] / (EnuMax - EnuMin)
    return RY_matrix


@lru_cache(maxsize=2**12)
def get_deltahs(cos_theta, hadr='SIBYLL2.3c'):
    MCEQ.set_interaction_model(hadr)
    MCEQ.set_theta_deg(np.degrees(np.arccos(cos_theta)))

    Xvec = np.logspace(np.log10(1),
                       np.log10(MCEQ.density_model.max_X), 10)
    heights = MCEQ.density_model.s_lX2h(np.log(Xvec)) * Units.cm
    deltahs = heights[:-1] - heights[1:]
    MCEQ.solve(int_grid=Xvec, grid_var="X")
    return deltahs


def categ_to_mothers(categ, daughter):
    charge = '-' if 'anti' in daughter else '+'
    bar = '-bar' if 'anti' in daughter else ''
    if categ == 'conv':
        mothers = ['pi'+charge, 'K'+charge, 'K0L']
    elif categ == 'pr':
        mothers = ['D'+charge, 'Ds'+charge, 'D0'+bar]
    else:
        mothers = [categ,]
    return mothers
    

def passing_rate(enu, cos_theta, kind='conv_numu', hadr='SIBYLL2.3c', accuracy=5, fraction=True):
    def get_rescale_phi(mother, deltah, idx):
        inv_decay_length_array = (
            ParticleProperties.mass_dict[mother] / MCEQ.e_grid * Units.GeV) *(
            deltah / ParticleProperties.lifetime_dict[mother])
        rescale_phi = inv_decay_length_array * MCEQ.get_solution(mother, grid_idx=idx)
        return interpolate.interp1d(MCEQ.e_grid, rescale_phi, fill_value='extrapolate')

    def get_integrand(categ, daughter, deltah, idx, weight_fn, esamp):
        mothers = categ_to_mothers(categ, daughter)
        ys = np.zeros(len(esamp))
        for mother in mothers:
            dNdEE = get_dNdEE(mother, daughter)[-1]
            rescale_phi = get_rescale_phi(mother, deltah, idx)
            ys += dNdEE(enu/esamp)/esamp*rescale_phi(esamp)*weight_fn(esamp)
            
        return ys

    categ, daughter = kind.split('_')
    
    ice_distance = overburden(cos_theta)
    identity = lambda Ep: 1
    reaching = lambda Ep: 1. - muon_reach_prob((Ep - enu) * Units.GeV, ice_distance)

    deltahs = get_deltahs(cos_theta, hadr)
    passing_numerator = 0
    passing_denominator = 0
    esamp = np.logspace(np.log10(enu), np.log10(MCEQ.e_grid[-1]), int(10**accuracy))
    for idx, deltah in enumerate(deltahs):
        passing_numerator += integrate.trapz(get_integrand(categ, daughter, deltah, idx, reaching, esamp), esamp)
        passing_denominator += integrate.trapz(get_integrand(categ, daughter, deltah, idx, identity, esamp), esamp)
        # print passing_numerator, passing_denominator
    return passing_numerator/passing_denominator if fraction else passing_numerator


def GetPassingFractionPrompt(cos_theta):
    caca = cs.CorrelatedSelfVetoProbabilityCalculator(cos_theta)
    e_grid = caca.mceq_run.e_grid
    delta = caca.mceq_run.e_widths
    ihijo = 20
    x_range = (e_grid / e_grid[ihijo])**-1
    dNdEE = caca.mceq_run.ds.get_d_matrix(411, 14)[ihijo] * e_grid / delta
    end_value = dNdEE[(x_range <= 1.) & (x_range >= 1.0e-3)][-1]
    dNdEE_DInterpolator = interpolate.interp1d(x_range[(x_range <= 1.) & (x_range >= 1.e-3)],
                                               dNdEE[(x_range <= 1.) & (
                                                   x_range >= 1.e-3)],
                                               bounds_error=False, fill_value=(end_value, 0.0))

    print "Calculating rescaled prompt yield"
    DToNeutrinoYield = GetRescaledYields(cos_theta, dNdEE_DInterpolator)
    rescale_prompt_decay_matrix = GetReachingRescaledYields(
        cos_theta, dNdEE_DInterpolator)

    passing_numerator = np.zeros(len(caca.mceq_run.e_grid))
    passing_denominator = np.zeros(len(caca.mceq_run.e_grid))

    for idx, XX in enumerate(caca.Xvec):
        if(idx >= len(caca.Xvec) - 1):
            continue
        height = caca.mceq_run.density_model.s_lX2h(
            np.log(caca.Xvec[idx])) * Units.cm
        deltah = (caca.mceq_run.density_model.s_lX2h(np.log(caca.Xvec[idx])) -
                  caca.mceq_run.density_model.s_lX2h(np.log(caca.Xvec[idx + 1]))) * Units.cm
        # do prompt
        inv_decay_length_array = (
            caca.ParticleProperties.mass_dict["D"] / caca.mceq_run.e_grid * Units.GeV) * (
            deltah / caca.ParticleProperties.lifetime_dict["D"])
        rescale_phi = inv_decay_length_array * \
            caca.mceq_run.get_solution("D-", 0, idx)

        passing_numerator += (np.dot(rescale_prompt_decay_matrix, rescale_phi))
        passing_denominator += (np.dot(DToNeutrinoYield, rescale_phi))
    return passing_numerator / passing_denominator
