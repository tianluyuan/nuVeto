import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from MCEq.core import MCEqRun
from MCEq.data import HadAirCrossSections
import CRFluxModels as pm
from mceq_config import config, mceq_config_without

r_dict ={}; mass_dict = {}; lifetime_dict = {}; a = {}; b = {}; pdg_id = {}; air_xs_inter = {};

# Carlos comment: all the units in the notebook should be converted to either
# GeV or inverse GeV by means of the following unit conversion factors.

# units
km = 5.0677309374099995 # km to GeV^-1 value from SQuIDS
gr = 5.62e+23 # gr to GeV value from SQuIDS
sec = 1523000.0 #$ sec to GeV^-1 from SQuIDS
cm = km*1.e-5

x_max = 100*km
x_min = 0*km

meson_list = ["kaon","pion"]

r_dict["kaon"]=0.046
r_dict["pion"]=0.573

mass_dict["kaon"]=0.493677 # GeV
mass_dict["pion"]=0.139570 # GeV

lifetime_dict["kaon"]=1.2389e-8*sec # s converted to GeV^-1
lifetime_dict["pion"]=2.6033e-8*sec # s converted to GeV^-1

a["ice"]=0.249 # GeV/mwe
a["rock"]=0.221 # GeV/mwe
b["ice"]=0.422e-3 # 1/mwe
b["rock"]=0.531e-3 # 1/mwe

pdg_id["kaon"] = 321 # k+
pdg_id["pion"] = 211 # pi+

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

def GetRelativeContributions(mceq_run):
    total_numu = mceq_run.get_solution('total_numu', 0, grid_idx=0)
    pion_prob = mceq_run.get_solution('pi_numu', 0, grid_idx=0)/total
    kaon_prob = mceq_run.get_solution('kaon_numu', 0, grid_idx=0)/total
    return pion_prob,kaon_prob

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
    return np.exp(-column_density/(air_xs_inter[meson]*cm**2)/mass_dict[meson])

def MuonDistance(muon_energy, medium = "ice"):
    if not (medium in a) or not (medium in b):
        raise Exception("Medium energy losses for muons not found.")
    a_ = a[medium]
    b_ = b[medium]

    return np.log(1.+ muon_energy*b_/a_)/b_

def GetColumnDensity(distance):
    return 1.

def MuonReachProbability(muon_energy,distance, column_density):
    return 1.

def NeutrinoFromParentProbability(Enu,costh,h,meson):
    return 1.

def ParentProductionProbability(primary_energy,costh,h,meson):
    return 1.

def CorrelatedProbability(Enu,costh):
    # here we implement the master formulae
    cprob = 0;
    for meson in meson_list:
        kernel = lambda x,Emu,h: NeutrinoFromParentProbability(Enu,costh,h,meson)*\
                                 NoInteractionProbability(Emu+Enu,GetColumnDensity(x+h),meson)*\
                                 DecayProbability(Emu+Enu,x+h,meson)*\
                                 ParentProductionProbability(Emu+Enu,costh,meson,h+x,meson)*\
                                 MuonReachProbability(Emu,h)

        r = r_dict[meson]
        Emu_min = Enu*r/(1.-r)
        Emu_max = 1.e10 # GeV
        cprob += integrate.tplquad(kernel,
                                    hmin,hmax,
                                    lambda h: Emu_min, lambda h: Emu_max,
                                    lambda h,Emu: x_min, lambda h, Emu: x_max)
    return cprob

if __name__ == "__main__":
    mierda = GetRunMCLayeredMode(30.)
    seccion_de_choque = cs_db.get_cs(pdg_id['pion'])
    print seccion_de_choque
    seccion_de_choque = cs_db.get_cs(pdg_id['kaon'])
    print seccion_de_choque
