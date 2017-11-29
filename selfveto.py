import numpy as np
import tqdm
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without
import correlated_selfveto as cs
from external import elbert, selfveto
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


def get_dNdEE(mother, daughter):
    ihijo = 20
    e_grid = MCEQ.e_grid
    delta = MCEQ.e_widths
    x_range = (e_grid / e_grid[ihijo])**-1
    dN = MCEQ.ds.get_d_matrix(mother, daughter)[ihijo]
    dNdEE = dN*e_grid/delta
    end_value = dNdEE[(x_range <= 1.) & (x_range >= 1.0e-3)][-1]
    return dN, interpolate.interp1d(
        x_range[(x_range <= 1.) & (x_range >= 1.e-3)],
        dNdEE[(x_range <= 1.) & (x_range >= 1.e-3)],
        bounds_error=False, fill_value=(end_value, 0.0))


def get_reaching(costh, dNdEE_Interpolator):
    e_grid = MCEQ.e_grid
    ice_distance = overburden(costh)
    RY_matrix = np.zeros((len(e_grid), len(e_grid)))
    for ipadre in tqdm.tqdm(range(len(e_grid))):
        for ihijo in range(len(e_grid)):
            # print "doing " + str(ipadre) + " " + str(ihijo)
            if ihijo > ipadre:
                RY_matrix[ihijo][ipadre] = 0
                continue

            if ihijo == 0 and ipadre == 0:
                RY_matrix[ihijo][ipadre] = 0
                continue

            EnuMin = e_grid[ihijo - 1]
            EnuMax = e_grid[ihijo]
            EpMin = e_grid[ipadre - 1]
            EpMax = e_grid[ipadre]

            if ihijo == 0:
                EnuMin = 10.  # GeV

            if ipadre == 0:
                EpMin = 10.  # GeV

            RY_matrix[ihijo][ipadre] = integrate.dblquad(
                lambda Ep, Enu: (dNdEE_Interpolator(
                    Enu / Ep) / Ep) * (1. - muon_reach_prob((Ep - Enu) * Units.GeV, ice_distance)),
                EnuMin, EnuMax,
                lambda Enu: np.max([Enu, EpMin]), lambda Enu: EpMax, epsabs=0, epsrel=1.e-2
            )[0] / ((EnuMax - EnuMin))
    return RY_matrix


def passing_rate(costh, kind='numu'):
    dN_kaon, dNdEE_kaon = get_dNdEE(ParticleProperties.pdg_id['kaon'],
                                    ParticleProperties.pdg_id[kind])
    dN_pion, dNdEE_pion = get_dNdEE(ParticleProperties.pdg_id['pion'],
                                    ParticleProperties.pdg_id[kind])

    print "Calculating rescaled kaon yield"
    dN_kaon_reach = get_reaching(costh, dNdEE_kaon)
    print "Calculating rescaled pion yield"
    dN_pion_reach = get_reaching(costh, dNdEE_pion)

    MCEQ.set_theta_deg(np.degrees(np.arccos(costh)))

    passing_numerator = np.zeros(len(MCEQ.e_grid))
    passing_denominator = np.zeros(len(MCEQ.e_grid))

    Xvec = np.logspace(np.log10(1),
                       np.log10(MCEQ.density_model.max_X), 1000)
    heights = MCEQ.density_model.s_lX2h(np.log(Xvec)) * Units.cm
    deltahs = np.diff(heights)
    MCEQ.solve(int_grid=Xvec, grid_var="X")
    for idx, deltah in enumerate(deltahs):
        # do for kaon
        inv_decay_length_array = (
            ParticleProperties.mass_dict["kaon"] / MCEQ.e_grid * Units.GeV) *(
            deltah / ParticleProperties.lifetime_dict["kaon"])
        rescale_phi = inv_decay_length_array * \
            MCEQ.get_solution("K-", 0, idx)

        passing_numerator += (np.dot(dN_kaon_reach, rescale_phi))
        passing_denominator += (np.dot(dN_kaon, rescale_phi))

        # do for pion
        inv_decay_length_array = (
            ParticleProperties.mass_dict["pion"] / MCEQ.e_grid * Units.GeV) *(
            deltah / ParticleProperties.lifetime_dict["pion"])
        rescale_phi = inv_decay_length_array * \
            MCEQ.get_solution("K-", 0, idx)

        passing_numerator += (np.dot(dN_pion_reach, rescale_phi))
        passing_denominator += (np.dot(dN_pion, rescale_phi))
    return passing_numerator / passing_denominator


def GetPassingFractionPrompt(costh):
    caca = cs.CorrelatedSelfVetoProbabilityCalculator(costh)
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
    DToNeutrinoYield = GetRescaledYields(costh, dNdEE_DInterpolator)
    rescale_prompt_decay_matrix = GetReachingRescaledYields(
        costh, dNdEE_DInterpolator)

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
