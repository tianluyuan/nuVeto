import numpy as np
from nuVeto.external import helper as exthp
from nuVeto.external import selfveto as extsv
from nuVeto.selfveto import *


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
