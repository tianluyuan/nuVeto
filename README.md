[![PyPI version](https://img.shields.io/pypi/v/nuveto)](https://pypi.org/project/nuveto) [![Build Status](https://travis-ci.com/tianluyuan/nuVeto.svg?branch=master)](https://travis-ci.com/tianluyuan/nuVeto) [![Python versions](https://img.shields.io/pypi/pyversions/nuveto)](https://pypi.org/project/nuveto)

# nuVeto

This package calculates the effect of a detector veto on the high-energy atmospheric neutrino flux via detection of muons that reach the detector. The result calculated is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle.

![Fluxes](/paper/figs_for_readme/fluxes_100.png?raw=true)

## Getting started

### Installing

This now relies on the updated version of [MCEq](https://github.com/afedynitch/MCEq). For the legacy version that relies on [MCEq_classic](https://github.com/afedynitch/MCEq_classic) do `git checkout v1.5` and see the README.

```bash
pip install nuVeto
```

This will install [MCEq](https://github.com/afedynitch/MCEq) with MKL.

Extras are `pip install nuVeto[plotting, resources]` which will install some packages for plotting and generating muon reaching probabilities.

### Usage

The simplest way to run is

```python
from nuVeto.nuveto import passing
from nuVeto.utils import Units
import crflux.models as pm

enu = 1e5*Units.GeV
cos_theta = 0.5
pf = passing(enu, cos_theta, kind='conv nu_mu',
             pmodel=(pm.HillasGaisser2012, 'H3a'),
             hadr='SIBYLL2.3c', depth=1950*Units.m,
             density=('CORSIKA', ('SouthPole','December')))
```
where kind can be `(conv|pr|_parent_) nu_(e|mu)(bar)`

See `examples/plots.py` for more detailed examples.

## Building muon detection probabilities

![Pdet](/paper/figs_for_readme/prpl_step1000.png?raw=true)

To calculate the passing fraction requires knowing the muon detection pdf as a function of the overburden and energy of the muon at the surface. This is constructed from a convolution of the muon reaching probability and the detector response. The muon reaching probability is constructed from MMC simulations and is provided for propagation in ice in `resources/mu/mmc/ice.pklz`. The detector response probability must be defined in `resources/mu/pl.py` as a function of the muon energy (at detector). Then, construct the overall muon detection pdf and place it into the correct location.

```bash
cd nuVeto/resources/mu
./mu.py -o ../../prpl/mymudet.pkl --plight pl_step_1000 mmc/ice_allm97.pklz
```

To use the newly generated file, pass it as a string to the `prpl` argument.
```bash
passing(enu, cos_theta, prpl='mymudet')`.
```

## Contributers
_Carlos Arguelles, Sergio Palomares-Ruiz, Austin Schneider, Logan Wille, Tianlu Yuan_
