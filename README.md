[![PyPI version](https://img.shields.io/pypi/v/nuveto)](https://pypi.org/project/nuveto) [![Build Status](https://github.com/tianluyuan/nuVeto/actions/workflows/pytest.yml/badge.svg)](https://github.com/tianluyuan/nuVeto/actions) [![Python versions](https://img.shields.io/pypi/pyversions/nuveto)](https://pypi.org/project/nuveto)

# nuVeto

This package calculates the effect of a detector veto on the high-energy atmospheric neutrino flux via detection of muons that reach the detector. The result calculated is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle.

![Fluxes](/paper/figs_for_readme/fluxes_100.png?raw=true)

## Getting started
It is recommended to work within a Python virtual environment.

```
python3 -m venv vdir
source vdir/bin/activate
```

### Installing

```bash
pip install nuVeto
```

This will install `numpy`, `scipy` and [`MCEq`](https://github.com/afedynitch/MCEq).

As of v2.3.1 a suite of tests is also packaged. It uses [`pytest`](https://docs.pytest.org/en/stable/), which can be optionally installed and run as follows.

```bash
pip install nuVeto[testing]
pytest --pyargs nuVeto
```

Extras are `pip install nuVeto[plotting, resources]` which will install necessary packages for making plots and generating alternative detector response parameterizations (muon reaching probabilities).

Note that v2.0 and higher rely on the updated version of [MCEq](https://github.com/afedynitch/MCEq). For the legacy version that relies on [MCEq_classic](https://github.com/afedynitch/MCEq_classic) do `git checkout v1.5` and see the README.

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

To calculate the passing fraction requires knowing the muon detection probability as a function of the overburden and energy of the muon at the surface. This is constructed from a convolution of the muon reaching probability and the detector response. The scripts for generating the necessary files are not packaged but provided in the `resources/` directory, which can be obtained with a clone of this repository. They also require some extra dependencies, which can be installed with `pip install nuVeto[resources]`.

The muon reaching probability is constructed from MMC simulations and is provided for propagation in ice in `resources/mu/mmc/ice_(allm97|bb).pklz` for two different cross section parameterizations. The detector response probability must be defined in `resources/mu/pl.py` as a function of the muon energy (at detector). Then, construct the overall muon detection probability.

```bash
cd resources/mu
./mu.py -o mymudet.pkl --plight pl_step_1000 mmc/ice_allm97.pklz
```

To use the newly generated file, pass the stem without file extension as a string to the `prpl` argument.
```bash
passing(enu, cos_theta, prpl='mymudet')`.
```

## Contributers
_Carlos Arguelles, Sergio Palomares-Ruiz, Austin Schneider, Logan Wille, Tianlu Yuan_
