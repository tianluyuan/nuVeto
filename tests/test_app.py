import os
import pytest
from itertools import product
from importlib import resources
import numpy as np
from scipy import interpolate
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.nuveto import passing, fluxes, nuVeto
from nuVeto.utils import Geometry, Units, MuonProb, amu, mceq_categ_format
import crflux.models as pm


def test_categ():
    assert nuVeto.categ_to_mothers('conv', 'nu_mu') == [
        'pi+', 'K+', 'K_L0', 'mu-']
    assert nuVeto.categ_to_mothers('conv', 'nu_mubar') == [
        'pi-', 'K-', 'K_L0', 'mu+']
    assert nuVeto.categ_to_mothers('conv', 'nu_e') == [
        'pi+', 'K+', 'K_L0', 'K_S0', 'mu+']
    assert nuVeto.categ_to_mothers('pr', 'nu_mu') == ['D+', 'D_s+', 'D0']
    assert nuVeto.categ_to_mothers('pr', 'nu_mubar') == ['D-', 'D_s-', 'Dbar0']


def test_costh_effective():
    geom = Geometry(1950*Units.m)
    cosths = np.linspace(-1, 1, 100)
    assert np.all(geom.cos_theta_eff(cosths) >= cosths)

    center = Geometry(geom.r_E)
    assert np.all(center.cos_theta_eff(cosths) == np.ones(100))


def test_overburden():
    geom = Geometry(1950*Units.m)
    cosths = np.linspace(-1, 1, 100)
    assert np.all(np.diff(geom.overburden(cosths)) < 0)

    center = Geometry(geom.r_E)
    assert np.all(center.overburden(cosths) == geom.r_E/Units.m)


def test_pdet():
    l_ice = np.linspace(1000, 200000, 500)
    emui = np.logspace(3, 8, 500)*Units.GeV
    coords = np.stack(np.meshgrid(emui, l_ice), axis=-1)
    root, subdir, fpaths = next(
        os.walk(resources.files('nuVeto') / 'data' / 'prpl'))
    for fpath in fpaths:
        muprob = MuonProb(os.path.splitext(fpath)[0])
        pdets = muprob.prpl(coords)
        assert np.all(pdets >= 0) and np.all(pdets <= 1)


def test_edge():
    """ Test edge case where MCEq yields are all <= 0.
    """
    sv = nuVeto(0., pmodel=(pm.ZatsepinSokolskaya, 'pamela'), hadr='DPMJET-III-19.1',
                density=('MSIS00_IC', ('SouthPole', 'June')))
    _ = sv.get_rescale_phi('D-', 508.0218046913023, 14)
    assert not np.any(_[:, -1] > 0)


@pytest.mark.parametrize('cth', [0.1, 0.3, 0.8])
def test_pnmshower(cth):
    particle = 14
    ecrs = amu(particle)*np.logspace(3, 10, 15)
    sv = nuVeto(cth)
    nmu = np.asarray([sv.nmu(ecr, particle) for ecr in ecrs])
    assert not np.any(nmu < 0)


@pytest.mark.parametrize('enu,l_ice,mother',
                         product(np.logspace(3, 7, 5),
                                 np.linspace(1500, 100000, 5),
                                 'pi+ K+ K_L0 D+ D0 Ds+'.split()))
def test_pnmsib(enu, l_ice, mother):
    psibs = nuVeto.psib(l_ice, mother, enu, 3, 'ice_allm97_step_1')
    assert np.all(0 <= psibs) and np.all(psibs <= 1)


@pytest.mark.parametrize('cth', [0.1, 0.3, 0.8])
def test_elbert(cth):
    ens = np.logspace(2, 8.9, 50)
    mine = np.asarray(
        [passing(en, cth, kind='conv nu_mu', hadr='DPMJET-III-3.0.6',
                 pmodel=(pm.GaisserHonda, None), prpl=None, corr_only=True) for en in ens])
    emu = extsv.minimum_muon_energy(extsv.overburden(cth))
    theirs = exthp.corr('conv nu_mu')(ens, emu, cth)
    print('test_elbert', cth)
    assert np.all(np.abs(theirs-mine) < 0.022)


@pytest.mark.parametrize('cth', [0.1, 0.3, 0.8])
def test_nuflux(cth):
    sv = nuVeto(cth)
    sv.grid_sol()
    kinds = ['conv nu_mu', 'conv nu_e', 'pr nu_mu', 'pr nu_e']
    for kind in kinds:
        _c, _d = kind.split()
        # thres = 1e7 if _c == 'pr' else 1e6
        thres = 1e7
        ensel = (sv.mceq.e_grid > 1e2) & (sv.mceq.e_grid < thres)
        theirs = sv.mceq.get_solution(mceq_categ_format(kind))[ensel]
        mine = np.asarray([fluxes(en, cth, kind, corr_only=True)[1]
                          for en in sv.mceq.e_grid[ensel]])

        print(kind, cth, theirs/mine)
        if _c == 'conv':
            assert np.all(np.abs(theirs/mine - 1) < 0.2)
        else:
            assert np.all(np.abs(theirs/mine - 1) < 0.8)


@pytest.mark.parametrize('cth', [0.9, 1])
def test_nonneg(cth, capsys):
    with capsys.disabled():
        sv = nuVeto(cth, debug_level=2)
        enus = [6.2e6, 1e7]
        kinds = ['conv nu_mu', 'conv nu_e', 'pr nu_mu', 'pr nu_e']
        for enu, kind in product(enus, kinds):
            n, d = sv.get_fluxes(enu, kind)
            assert n > 0 and d > 0
