from collections import namedtuple
import numpy as np


# Global Barr parameter table format
# ParamInfo([(x_min, x_max, E_min, E_max), ...], relative error, pdg) | x is x_lab= E_pi/E, E
# projectile-air interaction energy
ParamInfo = namedtuple('ParamInfo', 'regions error pdg')
BARR = {
    'a': ParamInfo([(0.0, 0.5, 0.00, 8.0)], 0.1, 211),
    'b1': ParamInfo([(0.5, 1.0, 0.00, 8.0)], 0.3, 211),
    'b2': ParamInfo([(0.6, 1.0, 8.00, 15.0)], 0.3, 211),
    'c': ParamInfo([(0.2, 0.6, 8.00, 15.0)], 0.1, 211),
    'd1': ParamInfo([(0.0, 0.2, 8.00, 15.0)], 0.3, 211),
    'd2': ParamInfo([(0.0, 0.1, 15.0, 30.0)], 0.3, 211),
    'd3': ParamInfo([(0.1, 0.2, 15.0, 30.0)], 0.1, 211),
    'e': ParamInfo([(0.2, 0.6, 15.0, 30.0)], 0.05, 211),
    'f': ParamInfo([(0.6, 1.0, 15.0, 30.0)], 0.1, 211),
    'g': ParamInfo([(0.0, 0.1, 30.0, 1e11)], 0.3, 211),
    'h1': ParamInfo([(0.1, 1.0, 30.0, 500.)], 0.15, 211),
    'h2': ParamInfo([(0.1, 1.0, 500.0, 1e11)], 0.15, 211),
    'i': ParamInfo([(0.1, 1.0, 500.0, 1e11)], 0.122, 211),
    'w1': ParamInfo([(0.0, 1.0, 0.00, 8.0)], 0.4, 321),
    'w2': ParamInfo([(0.0, 1.0, 8.00, 15.0)], 0.4, 321),
    'w3': ParamInfo([(0.0, 0.1, 15.0, 30.0)], 0.3, 321),
    'w4': ParamInfo([(0.1, 0.2, 15.0, 30.0)], 0.2, 321),
    'w5': ParamInfo([(0.0, 0.1, 30.0, 500.)], 0.4, 321),
    'w6': ParamInfo([(0.0, 0.1, 500., 1e11)], 0.4, 321),
    'x': ParamInfo([(0.2, 1.0, 15.0, 30.0)], 0.1, 321),
    'y1': ParamInfo([(0.1, 1.0, 30.0, 500.)], 0.3, 321),
    'y2': ParamInfo([(0.1, 1.0, 500., 1e11)], 0.3, 321),
    'z': ParamInfo([(0.1, 1.0, 500., 1e11)], 0.122, 321),
    'ch_a': ParamInfo([(0.0, 0.1, 0., 1e11)], 0.1, 411),
    'ch_b': ParamInfo([(0.1, 1.0, 0., 1e11)], 0.7, 411),
    'ch_e': ParamInfo([(0.1, 1.0, 800., 1e11)], 0.25, 411)
}


def barr_unc(xmat, egrid, pname, value):
    """Implementation of hadronic uncertainties as in Barr et al. PRD 74 094009 (2006)

    The names of parameters are explained in Fig. 2 and Fig. 3 in the paper."""
    # Energy dependence
    u = lambda E, val, ethr, maxerr, expected_err: val*min(
        maxerr/expected_err,
        0.122/expected_err*np.log10(E / ethr)) if E > ethr else 0.

    modmat = np.ones_like(xmat)
    modmat[np.tril_indices(xmat.shape[0], -1)] = 0.

    for minx, maxx, mine, maxe in BARR[pname].regions:
        eidcs = np.where((mine < egrid) & (egrid <= maxe))[0]
        for eidx in eidcs:
            xsel = np.where((xmat[:eidx + 1, eidx] >= minx) &
                            (xmat[:eidx + 1, eidx] <= maxx))[0]
            if not np.any(xsel):
                continue
            if pname in ['i', 'z']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 500., 0.5, 0.122)
            elif pname in ['ch_e']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 800., 0.3, 0.25)
            else:
                modmat[xsel, eidx] += value

    return modmat
