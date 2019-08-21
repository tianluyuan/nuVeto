import os
from pkg_resources import resource_filename
import numpy as np
from scipy import interpolate
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.nuveto import passing, fluxes, nuVeto
from nuVeto.utils import Geometry, Units, amu, MuonProb, old_categ_format
import crflux.models as pm


def test_categ():
    assert nuVeto.categ_to_mothers('conv', 'nu_mu') == ['pi+', 'K+', 'K_L0', 'mu-']
    assert nuVeto.categ_to_mothers('conv', 'nu_mubar') == ['pi-', 'K-', 'K_L0', 'mu+']
    assert nuVeto.categ_to_mothers('conv', 'nu_e') == ['pi+', 'K+', 'K_L0', 'K_S0', 'mu+']
    assert nuVeto.categ_to_mothers('pr', 'nu_mu') == ['D+', 'D_s+', 'D0']
    assert nuVeto.categ_to_mothers('pr', 'nu_mubar') == ['D-', 'D_s-', 'Dbar0']


def test_costh_effective():
    geom = Geometry(1950*Units.m)
    cosths = np.linspace(-1,1,100)
    assert np.all(geom.cos_theta_eff(cosths)>=cosths)

    center = Geometry(geom.r_E)
    assert np.all(center.cos_theta_eff(cosths) == np.ones(100))


def test_overburden():
    geom = Geometry(1950*Units.m)
    cosths = np.linspace(-1,1,100)
    assert np.all(np.diff(geom.overburden(cosths)) < 0)

    center = Geometry(geom.r_E)
    assert np.all(center.overburden(cosths) == geom.r_E/Units.m)


def test_pdet():
    l_ice = np.linspace(1000, 200000, 500)
    emui = np.logspace(3, 8, 500)*Units.GeV
    coords = np.stack(np.meshgrid(emui, l_ice), axis=-1)
    root, subdir, fpaths = os.walk(resource_filename('nuVeto','data/prpl/')).next()
    for fpath in fpaths:
        muprob = MuonProb(os.path.splitext(fpath)[0])
        pdets = muprob.prpl(coords)
        assert np.all(pdets >=0) and np.all(pdets <=1)
    

def test_pnmshower():
    cths = [0.1, 0.3, 0.8]
    particle = 14
    ecrs = amu(particle)*np.logspace(3, 10, 20)
    ecrs_fine = amu(particle)*np.logspace(3, 10, 1000)
    for cth in cths:
        sv = nuVeto(cth)
        nmu = [sv.nmu(ecr, particle) for ecr in ecrs]
        nmufn = interpolate.interp1d(ecrs, nmu, kind='linear',
                                     assume_sorted=True, fill_value=(0,np.nan))
        pnmshower = np.exp(-nmufn(ecrs_fine))
        assert np.all(0 <= pnmshower) and np.all(pnmshower <= 1)


def test_pnmsib():
    enus = np.logspace(3, 7, 5)
    l_ices = np.linspace(1500, 100000, 5)
    mothers = 'pi+ K+ K_L0 D+ D0 Ds+'.split()
    for enu in enus:
        for l_ice in l_ices:
            for mother in mothers:
                psibs = nuVeto.psib(l_ice, mother, enu, 3, 'ice_allm97_step_1')
                assert np.all(0 <= psibs) and np.all(psibs <= 1)


def test_elbert():
    ens = np.logspace(2,9,50)
    cths = [0.1,0.3,0.8]
    for cth in cths:
        mine = np.asarray(
            [passing(en, cth, kind='conv nu_mu', hadr='DPMJET-III-3.0.6',
                     pmodel=(pm.GaisserHonda, None), prpl=None, corr_only=True) for en in ens])
        emu = extsv.minimum_muon_energy(extsv.overburden(cth))
        theirs = exthp.corr('conv nu_mu')(ens, emu, cth)
        assert np.all(np.abs(theirs-mine)<0.022)


def test_nuflux():
    cths = [0.1, 0.3, 0.8]
    kinds = ['conv nu_mu', 'conv nu_e', 'pr nu_mu', 'pr nu_e']
    for cth in cths:
        sv = nuVeto(cth)
        sv.grid_sol()
        for kind in kinds:
            # _c, _d = kind.split()
            # thres = 1e7 if _c == 'pr' else 1e6
            thres = 1e7
            ensel = (sv.mceq.e_grid > 1e2) & (sv.mceq.e_grid < thres)
            theirs = sv.mceq.get_solution(old_categ_format(kind))[ensel]
            mine = np.asarray([fluxes(en, cth, kind, corr_only=True)[1] for en in sv.mceq.e_grid[ensel]])

            print kind, cth, theirs/mine
            assert np.all(np.abs(theirs/mine - 1) < 0.09)


def test_nonneg():
    cths = [0.9, 1]
    enus = [6.2e6, 1e7]
    kinds = ['conv nu_mu', 'conv nu_e', 'pr nu_mu', 'pr nu_e']
    for cth in cths:
        sv = nuVeto(cth)
        for kind in kinds:
            for enu in enus:
                n, d = sv.get_fluxes(enu, kind)
                assert n > 0 and d > 0
