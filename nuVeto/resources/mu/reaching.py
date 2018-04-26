#!/usr/bin/env python
"""
Generates initial muon energies and overburdens for input to MMC

Run as
./reaching.py | ./ammc -raw -user -sdec -lpm -bs=1 -ph=3 -bb=2 -sh=2 -scat -medi=ice -frho -f -r -vcut=1e-3 -cont > reaching.txt &
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
        for i in xrange(nsamples):
            print emui, distance
