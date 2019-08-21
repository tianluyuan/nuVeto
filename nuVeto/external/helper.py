import numpy as np
from functools import partial
from nuVeto.external import selfveto


def joint_passing_rate(enu, emu, cth, kind='numu'):
    return selfveto.correlated_passing_rate(enu, emu, cth)*selfveto.uncorrelated_passing_rate(enu, emu, cth, kind=kind)


def passrates(kind='conv nu_mu'):
    find_pf = {'conv nu_mu':partial(joint_passing_rate, kind='numu'),
               'pr nu_mu':partial(joint_passing_rate, kind='charm'),
               'conv nu_e':partial(selfveto.uncorrelated_passing_rate, kind='nue'),
               'pr nu_e':partial(selfveto.uncorrelated_passing_rate, kind='charm')}
    return find_pf[kind.replace('bar', '')]


def corr(kind='conv nu_mu'):
    find_pf = {'conv nu_mu':selfveto.correlated_passing_rate,
               'pr nu_mu':selfveto.correlated_passing_rate}
    return find_pf[kind.replace('bar', '')]


def uncorr(kind='conv nu_mu'):    
    find_pf = {'conv nu_mu':partial(selfveto.uncorrelated_passing_rate, kind='numu'),
               'pr nu_mu':partial(selfveto.uncorrelated_passing_rate, kind='charm'),
               'conv nu_e':partial(selfveto.uncorrelated_passing_rate, kind='nue'),
               'pr nu_e':partial(selfveto.uncorrelated_passing_rate, kind='charm')}
    return find_pf[kind.replace('bar', '')]
    
