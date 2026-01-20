from importlib import resources
from itertools import product
from pathlib import Path

import crflux.models as pm
import numpy as np
import pytest

from nuVeto import fluxes, nuVeto, passing
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.mu import MuonProb, interp
from nuVeto.utils import (
    Geometry,
    ParticleProperties,
    Units,
    amu,
    calc_bins,
    mceq_categ_format,
)


def test_calc_bins():
    bins = calc_bins(np.random.uniform(size=100))
    assert bins.min() > 0
    assert bins.max() < 3
    assert np.all(np.ediff1d(bins) > 0)


def test_interp():
    prpl = interp("ice_allm97", lambda emu: np.heaviside(emu - 1000, 1))

    geo = Geometry(1950*Units.m)
    psib = nuVeto.psib(geo.overburden(0.3), 'pi+', 1e5*Units.GeV, 3.5, prpl)
    pref = nuVeto.psib(geo.overburden(0.3), 'pi+', 1e5*Units.GeV, 3.5, 'ice_allm97_step_1')

    assert np.all(np.isclose(psib, pref))


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
    for fpath in (resources.files('nuVeto') / 'data' / 'prpl').iterdir():
        muprob = MuonProb(Path(fpath).stem)
        pdets = muprob.prpl(coords)
        assert np.all(pdets >= 0) and np.all(pdets <= 1)


def test_edge():
    """ Test edge case where MCEq yields are all <= 0.
    """
    sv = nuVeto(0., pmodel=(pm.ZatsepinSokolskaya, 'pamela'), hadr='DPMJET-III-19.1',
                density=('MSIS00_IC', ('SouthPole', 'June')))
    _ = sv.get_rescale_phi('D-', 508.0218046913023, 14)
    assert not np.any(_[:, -1] > 0)


def test_projectiles():
    projs = nuVeto(1.).projectiles()
    assert len(projs) == len(set(projs))
    assert set(projs) == set({'K+',
                              'K-',
                              'K_L0',
                              'K_S0',
                              'Lambda0',
                              'Lambdabar0',
                              'n0',
                              'nbar0',
                              'p+',
                              'pbar-',
                              'pi+',
                              'pi-'})
    assert set(projs) < set(ParticleProperties.pdg_id.keys())

    for hadr in ['DPMJETIII191',
                 'DPMJETIII306',
                 'EPOSLHC',
                 'QGSJET01C',
                 'QGSJETII03',
                 'QGSJETII04',
                 'SIBYLL21',
                 'SIBYLL23',
                 'SIBYLL23C',
                 'SIBYLL23C03',
                 'SIBYLL23CPP',
                 ]:
        assert set(projs) < set(nuVeto(1., hadr=hadr).mceq.pman.pname2pref.keys())


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
                                 'mu+ pi+ K+ K_L0 D+ D0 D_s+'.split()))
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
    kinds = [f'{_c} {_n}{_b}' for _c, _n, _b in
             product(['conv', 'pr'], ['nu_mu', 'nu_e'], ['', 'bar'])]
    for kind in kinds:
        ensel = (sv.mceq.e_grid > 1e3) & (sv.mceq.e_grid < 1e7)
        theirs = sv.mceq.get_solution(mceq_categ_format(kind))[ensel]
        mine = np.asarray([fluxes(en, cth, kind, corr_only=True)[1]
                          for en in sv.mceq.e_grid[ensel]])

        print(kind, cth, theirs/mine)
        assert np.all(np.abs(theirs/mine - 1) < 0.2)


@pytest.mark.parametrize('cth', [0.9, 1])
def test_nonneg(cth, capsys):
    with capsys.disabled():
        sv = nuVeto(cth, debug_level=2)
        enus = [6.2e6, 1e7]
        kinds = [f'{_c} {_n}{_b}' for _c, _n, _b in
                 product(['conv', 'pr'], ['nu_mu', 'nu_e'], ['', 'bar'])]
        for enu, kind in product(enus, kinds):
            n, d = sv.get_fluxes(enu, kind)
            assert n > 0 and d > 0
