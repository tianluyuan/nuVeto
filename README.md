# nuVeto

This package calculates the effect of a detector veto on the high-energy atmospheric neutrino flux via detection of muons that reach the detector. The result calculated is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle. 

## Getting started

### Prerequisites

The package relies on [MCEq](https://github.com/afedynitch/MCEq/) which in turn depends on some optimized python libraries. These libraries can be installed with [Anaconda](http://continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) following the instructions to install MCEq.

### Installing

To install directly
```
pip install git+ssh://git@github.com/arguelles/nuVeto#egg=nuVeto
```

Or if you prefer to clone the repository
```
git clone https://github.com/arguelles/nuVeto
cd nuVeto
pip install -e .
```

### Usage

Some examples are in the `tests/` and `examples/` directories. The simplest way to run is

```
from nuVeto.selfveto import *
enu = 1e6
cos_theta = 0.5
pf = passing(enu, cos_theta, kind='conv_numu',
             pmodel=(pm.HillasGaisser2012, 'H3a'),
             hadr='SIBYLL2.3c', depth=1950*Units.m)
```

## Tests
