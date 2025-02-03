#!/usr/bin/env python
"""
Generates initial muon energies and overburdens for input to MMC

Run as
./reaching.py | ./ammc -raw -user -sdec -lpm -bs=1 -ph=3 -bb=2 -sh=2 -scat -medi=ice -frho -f -r -vcut=1e-3 -cont > reaching.txt &

From https://icecube.wisc.edu/~dima/work/MUONPR/MUON/WEBMMC/HISTORY.TXT
    "Several new cross sections are implemented: Andreev, Bezrukov, and Bugaev
    (abb) parameterization of the bremsstrahlung cross section can now be selected
    with -bs=2. The previously available kkp (Kelner, Kokoulin, and Petrukhin) can
    be chosen with -bs=1.

    The options -allm and -phnu are now obsolete. A new option -ph=[1-4] chooses
    Bezrukov and Bugaev photonuclear cross section parameterization (1), Bezrukov
    and Bugaev with hard component from Bugaev and Shlepin (2), ALLM (3), and
    Butkevich and Mikheyev (4) parameterization. Another option -bb (similar to
    the former -phnu) chooses one of the parameterizations of the soft component
    in the BB cross section (for -ph=1-2, values of -bb=3/4 correspond to BB/ZEUS
    photon-nucleon cross section formula) or version of the ALLM parameterization
    (for -ph=3, values of -bb=1/2 choose the ALLM91/ALLM97 parameter sets).

    New option -sh=[1-2] is introduced. It allows to choose the nuclear structure
    function calculation algorithm for -ph=[3-4] according to Dutta-Seckel or
    Butkevich-Mikheyev papers.

    Hadronic channels of the tau decay are now described as two-body decays into a
    tau neutrino and a pion or a rho-770, a1-1260, or rho-1450 resonance."
"""
import numpy as np

nsamples = 1000
xvec = np.logspace(np.log10(1./1000), np.log10(1./(2e5)), 100)
emuis = np.logspace(2, 8, 100)

for distance in 1/xvec:
    for emui in emuis:
        # roughly assume 1GeV/4m minimum energy loss
        if emui < distance/4.:
            continue
        for i in range(nsamples):
            print(emui, distance)
