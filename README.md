# nuVeto
[![Build Status](https://travis-ci.com/tianluyuan/nuVeto.svg?branch=master)](https://travis-ci.com/tianluyuan/nuVeto)

This package calculates the effect of a detector veto on the high-energy atmospheric neutrino flux via detection of muons that reach the detector. The result calculated is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle. 

## Getting started

### Prerequisites

The package relies on [MCEq](https://github.com/afedynitch/MCEq/) which in turn depends on some optimized python libraries. These libraries can be installed with [Anaconda](http://continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) following the instructions to install MCEq.

`matplotlib` and `pandas` are optional packages for plotting and generating muon reaching probabilities.

The code was tested against MCEq commit `5757d0b`.

### Installing

To install directly
```bash
pip install git+git://git@github.com/tianluyuan/nuVeto#egg=nuVeto
```

Or if you prefer to clone the repository
```bash
git clone https://github.com/tianluyuan/nuVeto
cd nuVeto
pip install -e .
```

### Usage

The simplest way to run is

```python
from nuVeto.nuveto import passing
enu = 1e5*Units.GeV
cos_theta = 0.5
pf = passing(enu, cos_theta, kind='conv_numu',
             pmodel=(pm.HillasGaisser2012, 'H3a'),
             hadr='SIBYLL2.3c', depth=1950*Units.m,
             density=('CORSIKA', ('SouthPole','June')))
```

Running with `'MSIS00'` density models in c-mode requires running `make` in `MCEq/c-NRLMSISE-00`. See the `examples/` directory for more detailed examples.

## Building muon detection probabilities

To calculate the passing fraction requires knowing the muon detection pdf as a function of the overburden and energy of the muon at the surface. This is constructed from a convolution of the muon reaching probability and the detector response. The muon reaching probability is constructed from MMC simulations and is provided for propagation in ice in `resources/mu/mmc/ice.pklz`. The detector response probability must be defined in `resources/mu/pl.py` as a function of the muon energy (at detector). Then, construct the overall muon detection pdf and place it into the correct location.

```bash
cd nuVeto/resources/mu
./mu.py -o ../../prpl/mymudet.pkl --plight pl_step_1000 mmc/ice_allm97.pklz
```

To use the newly generated file, pass it as a string to the `prpl` argument.
```bash
passing(enu, cos_theta, prpr='mymudet')`.
```

## Contributers
_Carlos Arguelles, Sergio Palomares-Ruiz, Austin Schneider, Logan Wille, Tianlu Yuan_
