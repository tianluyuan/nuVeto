import numpy as np
import tqdm
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import correlated_selfveto as cs
import utils
from utils import Units
from external import elbert, selfveto


def GetReachingRescaledYields(costh, dNdEE_Interpolator, e_grid, caca):
    ice_distance = utils.overburden(costh)
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
                    Enu / Ep) / Ep) * (1. - caca.MuonReachProbability((Ep - Enu) * Units.GeV, ice_distance)),
                EnuMin, EnuMax,
                lambda Enu: np.max([Enu, EpMin]), lambda Enu: EpMax, epsabs=0, epsrel=1.e-2
            )[0] / ((EnuMax - EnuMin))
    return RY_matrix


def GetRescaledYields(costh, dNdEE_Interpolator, e_grid):
    ice_distance = utils.overburden(costh)
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
                lambda Ep, Enu: (dNdEE_Interpolator(Enu / Ep) / Ep),
                EnuMin, EnuMax,
                lambda Enu: np.max([Enu, EpMin]), lambda Enu: EpMax, epsabs=0, epsrel=1.e-2
            )[0] / ((EnuMax - EnuMin))
    return RY_matrix


def GetPassingFraction(costh):
    caca = cs.CorrelatedSelfVetoProbabilityCalculator(costh)
    e_grid = caca.mceq_run.e_grid
    delta = caca.mceq_run.e_widths
    ihijo = 20
    x_range = (e_grid / e_grid[ihijo])**-1
    dNdEE = caca.mceq_run.ds.get_d_matrix(
        caca.ParticleProperties.pdg_id["kaon"], 14)[ihijo] * e_grid / delta
    end_value = dNdEE[(x_range <= 1.) & (x_range >= 1.0e-3)][-1]
    dNdEE_KaonInterpolator = interpolate.interp1d(x_range[(x_range <= 1.) & (x_range >= 1.e-3)],
                                                  dNdEE[(x_range <= 1.) & (
                                                      x_range >= 1.e-3)],
                                                  bounds_error=False, fill_value=(end_value, 0.0))

    x_range = (e_grid / e_grid[ihijo])**-1
    dNdEE = caca.mceq_run.ds.get_d_matrix(
        caca.ParticleProperties.pdg_id["pion"], 14)[ihijo] * e_grid / delta
    end_value = dNdEE[(x_range <= 1.) & (x_range >= 1.0e-3)][-1]
    dNdEE_PionInterpolator = interpolate.interp1d(x_range[(x_range <= 1.) & (x_range >= 1.e-3)],
                                                  dNdEE[(x_range <= 1.) & (
                                                      x_range >= 1.e-3)],
                                                  bounds_error=False, fill_value=(end_value, 0.0))

    print "Calculating rescaled kaon yield"
    # KaonToNeutrinoYield=caca.mceq_run.ds.get_d_matrix(caca.ParticleProperties.pdg_id["kaon"],14)
    KaonToNeutrinoYield = GetRescaledYields(costh, dNdEE_KaonInterpolator, e_grid)
    rescale_kaon_decay_matrix = GetReachingRescaledYields(
        costh, dNdEE_KaonInterpolator, e_grid, caca)
    print "Calculating rescaled pion yield"
    # PionToNeutrinoYield=caca.mceq_run.ds.get_d_matrix(caca.ParticleProperties.pdg_id["pion"],14)
    PionToNeutrinoYield = GetRescaledYields(costh, dNdEE_PionInterpolator, e_grid)
    rescale_pion_decay_matrix = GetReachingRescaledYields(
        costh, dNdEE_PionInterpolator, e_grid, caca)

    passing_guy_numerator = np.zeros(len(caca.mceq_run.e_grid))
    passing_guy_denominator = np.zeros(len(caca.mceq_run.e_grid))

    for idx, XX in enumerate(caca.Xvec):
        if(idx >= len(caca.Xvec) - 1):
            continue
        height = caca.mceq_run.density_model.s_lX2h(
            np.log(caca.Xvec[idx])) * Units.cm
        deltah = (caca.mceq_run.density_model.s_lX2h(np.log(caca.Xvec[idx])) -
                  caca.mceq_run.density_model.s_lX2h(np.log(caca.Xvec[idx + 1]))) * Units.cm
        # if deltah>xmax:
        #    xmax=deltah
        # do for kaon
        inv_decay_length_array = (
            caca.ParticleProperties.mass_dict["kaon"] / caca.mceq_run.e_grid * Units.GeV) * (
            deltah / caca.ParticleProperties.lifetime_dict["kaon"])
        rescale_phi = inv_decay_length_array * \
            caca.mceq_run.get_solution("K-", 0, idx)

        passing_guy_numerator += (np.dot(rescale_kaon_decay_matrix, rescale_phi))
        passing_guy_denominator += (np.dot(KaonToNeutrinoYield, rescale_phi))

        # do for pion
        inv_decay_length_array = (
            caca.ParticleProperties.mass_dict["pion"] / caca.mceq_run.e_grid * Units.GeV) * (
            deltah / caca.ParticleProperties.lifetime_dict["pion"])
        rescale_phi = inv_decay_length_array * \
            caca.mceq_run.get_solution("pi-", 0, idx)

        passing_guy_numerator += (np.dot(rescale_pion_decay_matrix, rescale_phi))
        passing_guy_denominator += (np.dot(PionToNeutrinoYield, rescale_phi))
    return passing_guy_numerator / passing_guy_denominator


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

    passing_guy_numerator = np.zeros(len(caca.mceq_run.e_grid))
    passing_guy_denominator = np.zeros(len(caca.mceq_run.e_grid))

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

        passing_guy_numerator += (np.dot(rescale_prompt_decay_matrix, rescale_phi))
        passing_guy_denominator += (np.dot(DToNeutrinoYield, rescale_phi))
    return passing_guy_numerator / passing_guy_denominator
