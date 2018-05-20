import numpy as np
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import passing, fluxes, SelfVeto
from nuVeto.utils import Geometry, Units
try:
    import CRFluxModels.CRFluxModels as pm
except ImportError:
    import CRFluxModels as pm


def test_is_prompt():
    assert SelfVeto.is_prompt('pr')
    assert not SelfVeto.is_prompt('conv')


def test_categ():
    assert SelfVeto.categ_to_mothers('conv', 'numu') == ['pi+', 'K+', 'K0L', 'mu-']
    assert SelfVeto.categ_to_mothers('conv', 'antinumu') == ['pi-', 'K-', 'K0L', 'mu+']
    assert SelfVeto.categ_to_mothers('conv', 'nue') == ['pi+', 'K+', 'K0L', 'K0S', 'mu+']
    assert SelfVeto.categ_to_mothers('pr', 'numu') == ['D+', 'Ds+', 'D0']
    assert SelfVeto.categ_to_mothers('pr', 'antinumu') == ['D-', 'Ds-', 'D0-bar']


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


def test_elbert():
    echoice = exthp.corr
    ens = np.logspace(2,9,50)
    cths = [0.1,0.3,0.8]
    for cth in cths:
        mine = np.asarray(
            [passing(en, cth, kind='conv_numu', hadr='DPMJET-III',
                     pmodel=(pm.GaisserHonda, None), prpl=None, corr_only=True) for en in ens])
        emu = extsv.minimum_muon_energy(extsv.overburden(cth))
        theirs = exthp.corr('conv_numu')(ens, emu, cth)
        assert np.all(np.abs(theirs-mine)<0.02)


def test_nuflux():
    cths = [0.1, 0.3, 0.8]
    kinds = ['conv_numu', 'conv_nue', 'pr_numu', 'pr_nue']
    for cth in cths:
        sv = SelfVeto(cth)
        sv.grid_sol()
        for kind in kinds:
            thres = 1e7 if sv.is_prompt(kind) else 1e6
            ensel = (sv.mceq.e_grid > 1e2) & (sv.mceq.e_grid < thres)
            theirs = sv.mceq.get_solution(kind)[ensel]
            mine = np.asarray([fluxes(en, cth, kind, corr_only=True)[1] for en in sv.mceq.e_grid[ensel]])

            print kind, cth, theirs/mine
            assert np.all(np.abs(theirs/mine - 1) < 0.09)
