# Copyright © 2017 C. Arguelles, S. Palomares-Ruiz, A. Schneider, T. Yuan, and L. Wille
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the “Software”), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# For more information please email:
#
# C. Arguelles (caad@mit.edu)
# S. Palomres-Ruiz (sergio.palomares.ruiz@ific.uv.es)
# A. Schneider (aschneider@icecube.wisc.edu)
# T. Yuan (tianlu.yuan@icecube.wisc.edu)
# L. Wille (lwille@icecube.wisc.edu)
#
# Please cite:
# arXiv:XXXX.XXXX

import numpy as np
import scipy as sp
import scipy.stats as stats
import math
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from MCEq.core import MCEqRun
from MCEq.data import HadAirCrossSections
import CRFluxModels as pm
import geometry as geo
from mceq_config import config, mceq_config_without

class SelfVetoProbabilityCalculator(object):
    class Units(object):
        # units
        km = 5.0677309374099995 # km to GeV^-1 value from SQuIDS
        gr = 5.62e+23 # gr to GeV value from SQuIDS
        sec = 1523000.0 #$ sec to GeV^-1 from SQuIDS
        cm = km*1.e-5
        GeV = 1
    class ParticleProperties(object):
        r_dict ={}; mass_dict = {}; lifetime_dict = {}; pdg_id = {}; air_xs_inter = {};

        _dict["kaon"]=0.046
        r_dict["pion"]=0.573

        mass_dict["kaon"]=0.493677 # GeV
        mass_dict["pion"]=0.139570 # GeV

        lifetime_dict["kaon"]=1.2389e-8*sec # s converted to GeV^-1
        lifetime_dict["pion"]=2.6033e-8*sec # s converted to GeV^-1

        pdg_id["kaon"] = 321 # k+
        pdg_id["pion"] = 211 # pi+

    class MaterialProperties(object):
        a = {}; b = {}; density = {};
        a["ice"]=0.249 # GeV/mwe
        a["rock"]=0.221 # GeV/mwe
        b["ice"]=0.422e-3 # 1/mwe
        b["rock"]=0.531e-3 # 1/mwe
        density["ice"] = 0.9167 # g/cm^3

    def  __init__(self,):
        pass

class CorrelatedSelfVetoProbabilityCalculator(SelfVetoProbabilityCalculator):
    def  __init__(self,):
        pass


x_max = 100*km
x_min = 0*km

d = 10*km

meson_list = ["kaon","pion"]

cs_db = HadAirCrossSections('SIBYLL2.1')

air_xs_inter["kaon"] = interpolate.interp1d(cs_db.egrid,cs_db.get_cs(pdg_id['kaon'])) # input GeV return cm^2
air_xs_inter["pion"] = interpolate.interp1d(cs_db.egrid,cs_db.get_cs(pdg_id['pion'])) # input GeV return cm^2

def GetRunMCLayeredMode(theta,hadronic_model='SIBYLL-2.3c',primary_model=(pm.HillasGaisser2012, 'H3a')):
    cfg = dict(config)
    mceq_run = MCEqRun(
                    hadronic_model,
                    primary_model=primary_model,
                    theta_deg=theta,
                    **cfg
                )
    Xvec = np.arange(1, mceq_run.density_model.max_X, 5)
    mceq_run.solve(int_grid=Xvec, grid_var="X")
    return mceq_run

mceq_run  = GetRunMCLayeredMode(30.)

def GetRelativeContributions(mceq_run):
    total_numu = mceq_run.get_solution('total_numu', 0, grid_idx=0)
    pion_prob = mceq_run.get_solution('pi_numu', 0, grid_idx=0)/total_numu
    kaon_prob = mceq_run.get_solution('k_numu', 0, grid_idx=0)/total_numu
    return pion_prob,kaon_prob

pion_prob,kaon_prob = GetRelativeContributions(mceq_run)

def FindNearest(array,value):
    return np.searchsorted(array, value, side="left")

def MinimumMuonBrotherEnergy(neutrino_energy,meson):
    """
    Returns the minimum muon energy of the brother muon.
    Eq. (5) from 0812.4308
    """
    if not(meson in r_dict):
        raise Exception("Meson not found in mass dictionary.")
    r = r_dict[meson]
    return neutrino_energy*r/(1.-r)

def MinimumMesonParentEnergy(neutrino_energy,meson):
    """
    Returns the minimum parent meson energy.
    Eq. (5) from 0812.4308
    """
    if not (meson in r_dict):
        raise Exception("Meson not found in mass dictionary.")
    r = r_dict[meson]
    return neutrino_energy/(1.-r)

def DecayProbability(primary_energy, distance, meson):
    if not (meson in r_dict):
        raise Exception("Meson not found lifetime dictionary.")
    boost_factor=primary_energy/mass_dict[meson]
    return np.exp(-distance/(boost_factor*lifetime_dict[meson]))

def NoInteractionProbability(primary_energy, column_density, meson):
    if not (meson in r_dict):
        raise Exception("Meson not found cross section dictionary.")
    return np.exp(-column_density/(air_xs_inter[meson](primary_energy)*cm**2)/mass_dict[meson])

def MeanMuonDistance(muon_energy, medium = "ice"):
    if not (medium in a) or not (medium in b):
        raise Exception("Medium energy losses for muons not found.")
    a_ = a[medium]
    b_ = b[medium]

    return np.log(1.+ muon_energy*b_/a_)/b_

def GetAirColumnDensity(height,distance):
    return (mceq_run.density_model.s_h2X(height) - mceq_run.density_model.s_h2X(height+distance))

def GetIceColumnDensity(costh, depth):
    return geo.overburden(costh, depth, elevation=2400)

def MuonReachProbability(muon_energy, height, ice_column_density):
    # simplifying assumption that the muon reach distribution is a gaussian
    return stats.norm.sf(ice_column_density/density["ice"],
            loc=MeanMuonDistance(muon_energy),scale=np.sqrt(MeanMuonDistance(muon_energy)))

def NeutrinoFromParentProbability(neutrino_energy,costh,h,meson):
    ie = FindNearest(mceq_run.cs.egrid,neutrino_energy/GeV)
    if meson == "pion":
        return pion_prob[ie]
    elif meson == "kaon":
        return pion_prob[ie]
    else:
        raise Exception("Invalid meson parent")

def ParentProductionProbability(primary_energy,costh,h,meson):
    ie = FindNearest(mceq_run.cs.egrid,primary_energy/GeV)
    if meson == "pion":
        return mceq_run.get_solution('pi-', 0, grid_idx=0)[ie]
    elif meson == "kaon":
        return mceq_run.get_solution('K-', 0, grid_idx=0)[ie]
    else:
        raise Exception("Invalid meson. ")

def CorrelatedProbability(Enu,costh):
    # here we implement the master formulae
    cprob = 0;
    for meson in meson_list:
        kernel = lambda x,Emu,h: NeutrinoFromParentProbability(Enu,costh,h,meson)*\
                                 DecayProbability(Emu+Enu,x+h,meson)*\
                                 NoInteractionProbability(Emu+Enu,GetAirColumnDensity(h,x),meson)*\
                                 ParentProductionProbability(Emu+Enu,costh,h+x,meson)*\
                                 MuonReachProbability(Emu,h,GetIceColumnDensity(costh,d))

        r = r_dict[meson]
        h_min = 0; h_max = 40;
        Emu_min = Enu*r/(1.-r)
        Emu_max = 1.e10 # GeV
        cprob += integrate.tplquad(kernel,
                                    h_min,h_max,
                                    lambda h: Emu_min, lambda h: Emu_max,
                                    lambda h,Emu: x_min, lambda h, Emu: x_max)[0]
    return cprob

if __name__ == "__main__":
    mierda = GetRunMCLayeredMode(30.)
    seccion_de_choque = cs_db.get_cs(pdg_id['pion'])
    print seccion_de_choque
    seccion_de_choque = cs_db.get_cs(pdg_id['kaon'])
    print seccion_de_choque
