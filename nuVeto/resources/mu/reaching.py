#!/usr/bin/env python
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
