import numpy as np
from functools import partial
import selfveto


def joint_passing_rate(enu, emu, cth, kind='numu'):
    return selfveto.correlated_passing_rate(enu, emu, cth)*selfveto.uncorrelated_passing_rate(enu, emu, cth, kind=kind)


def passrates(kind='conv_numu'):
    find_pf = {'conv_numu':partial(joint_passing_rate, kind='numu'),
               'pr_numu':partial(joint_passing_rate, kind='charm'),
               'conv_nue':partial(selfveto.uncorrelated_passing_rate, kind='nue'),
               'pr_nue':partial(selfveto.uncorrelated_passing_rate, kind='charm')}
    return find_pf[kind]


def uncorr(kind='conv_numu'):
    find_pf = {'conv_numu':partial(selfveto.uncorrelated_passing_rate, kind='numu'),
               'pr_numu':partial(selfveto.uncorrelated_passing_rate, kind='charm'),
               'conv_nue':partial(selfveto.uncorrelated_passing_rate, kind='nue'),
               'pr_nue':partial(selfveto.uncorrelated_passing_rate, kind='charm')}
    return find_pf[kind]
