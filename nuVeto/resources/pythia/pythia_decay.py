import pythia8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
pythia = pythia8.Pythia()

pythia.readString("ProcessLevel:all = off")
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 0")
pythia.init()

UnstableParents = {411: "D+", -411: "D-"}

for parent in UnstableParents:
    # make parents unstable
    pythia.particleData.mayDecay(parent, True)

StableDaughters = {-13: "mu+", 13: "mu-",
                   15: "tau-", -15: "tau+",
                   14: "numu", -14: "antinumu",
                   12: "nue", -12: "antinue",
                   16: "nutau", -16: "antinutau",
                   211: "pi+", -211: "pi-",
                   321: "k+", 321: "k-",
                   130: "k0l", 310: "k0s"}

GuysWeWant = {-13: "mu+", 13: "mu-",
              14: "numu", -14: "antinumu", }

for daughter in StableDaughters:
    # make daughters stable
    pythia.particleData.mayDecay(daughter, False)

number_of_decays = 1000000

resultsNuMu = {}
resultsMuons = {}

for parent in UnstableParents:
    mass = pythia.particleData.m0(parent)
    momenta = 1.e3  # GeV
    energy = np.sqrt(mass*mass + momenta*momenta)
    p4vec = pythia8.Vec4(momenta, 0., 0., energy)  # GeV

    resultsNuMu[parent] = []
    resultsMuons[parent] = []

    for i in range(number_of_decays):
        # clean previous stuff
        pythia.event.reset()
        # pdgid, status, col, acol, p, m
        # status = 91 : make normal decay products
        # col, acol : color and anticolor indices
        pythia.event.append(parent, 91, 0, 0, p4vec, mass)
        # go all the way
        pythia.forceHadronLevel()
        for j in range(pythia.event.size()):
            if (not pythia.event[j].isFinal()):
                # if not done decaying continue decaying
                continue
            if (pythia.event[j].id() in [-14, 14]):  # is neutrino type I care about
                for sister in pythia.event[j].sisterList():
                    if (pythia.event[sister].id() in [-13, 13]):  # is muon
                        resultsNuMu[parent].append(pythia.event[j].e()/energy)
                        resultsMuons[parent].append(
                            pythia.event[sister].e()/energy)

plt.figure(figsize=(6, 5))
H, xedges, yedges, fig = plt.hist2d(np.array(resultsNuMu[421]), np.array(resultsMuons[421]),
                                    bins=[np.arange(0., 1.001, 0.01), np.arange(0., 1.001, 0.01)])
plt.close()

H_norm_rows = H / H.max(axis=1, keepdims=True)

xcenters = xedges[:-1]+np.diff(xedges)
ycenters = yedges[:-1]+np.diff(yedges)

histogramas = []
plt.figure(figsize=(7, 5))
for i, row in enumerate(H_norm_rows):
    # if xcenters[i] in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    norm = np.sum(row)
    hist, bin_edges, fig = plt.hist(xcenters, bins=xedges, weights=row/norm,
                                    histtype="step", lw=2, label=r"$E_\nu/E_D =" + str(ycenters[i]) + "$")
    histogramas.append(hist)
histogramas = np.array(histogramas)

plt.legend()
plt.xlim(0, 1)
plt.xlabel(r"$E_\mu/E_D$", fontsize=20)
plt.close()

ceros = np.zeros(len(xcenters))
ceros[0] = 1.
reshape_hist = np.vstack([histogramas, ceros])

np.savez("DP_meson_decay_distributions_with_two_bodies",
         histograms=reshape_hist, xedges=xedges)
