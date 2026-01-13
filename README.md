[![PyPI version](https://img.shields.io/pypi/v/nuveto)](https://pypi.org/project/nuveto) [![Build Status](https://github.com/tianluyuan/nuVeto/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/tianluyuan/nuVeto/actions) [![Python versions](https://img.shields.io/pypi/pyversions/nuveto)](https://pypi.org/project/nuveto)

# nuVeto

This package calculates the effect of a detector veto, triggered by the accompanying muons from the same air shower, on the high-energy atmospheric neutrino flux. The calculated result is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle.

As a corollary, the total atmospheric neutrino flux as well as parent particle fluxes at various depths in the atmosphere are calculated using [`MCEq`](https://github.com/mceq-project/MCEq). To illustrate, the parent fluxes are accessible as shown in this [example](https://github.com/tianluyuan/nuVeto/blob/eb643373534c289ef11a01c43fc82bceeb76c122/src/nuVeto/examples/plots.py#L359).

<img width="49%" alt="fluxes" src="https://github.com/tianluyuan/nuVeto/blob/main/paper/figs_for_readme/fluxes_100.png?raw=true" > <img width="49%" alt="Ds+_025" src="https://github.com/tianluyuan/nuVeto/blob/main/paper/figs_for_readme/Ds+_0.25.png?raw=true" >

In the right panel, solid lines are direct calculations from `MCEq` while dashed lines are an extension down to lower energies where the parent particle is assumed to no longer interact and their fluxes are computed based on the CR flux, cross section, yield and parent's decay length.

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

This will install `numpy`, `scipy`, `pandas` and `MCEq`. By default this will not install MKL or CUDA libraries that accelerate `MCEq`. If your system has the right hardware available, it is possible get a substantial speed up with `pip install mkl` or `pip install MCEq[cuda]`. The fastest library is automatically chosen.

As of v2.3.1 a suite of tests is also packaged. It uses [`pytest`](https://docs.pytest.org/en/stable/), which can be optionally installed and run as follows.

```bash
pip install nuVeto[testing]
pytest --pyargs nuVeto
```

Additional optional dependencies are `[plotting]` and `[pythia8]` which will respectively install needed packages for making example plots (`from nuVeto.examples import plots`), and generating alternative hadron decay rates with PYTHIA.

Note that v2.0 and higher rely on the updated version of [MCEq](https://github.com/mceq-project/MCEq). For the legacy version that relies on [MCEq_classic](https://github.com/afedynitch/MCEq_classic) do `git checkout v1.5` and follow the instructions in the README.

### Usage

The simplest way to run is

```python
from nuVeto import passing
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

<img width="49%" alt="Pdet" src="https://github.com/tianluyuan/nuVeto/blob/main/paper/figs_for_readme/prpl_step1000.png?raw=true"> <img width="49%" alt="Pdet" src="https://github.com/tianluyuan/nuVeto/blob/main/paper/figs_for_readme/prpl_sigmoid.png?raw=true">

To calculate the passing fraction requires knowing the muon detection probability as a function of the overburden and energy of the muon at the surface. This is constructed from a convolution of the muon reaching probability and the detector response. The scripts for generating the necessary files are provided in the `scripts/mu/` subdirectory, which can be obtained with a download or clone of this repository.

The muon reaching probability is constructed from MMC simulations and is provided for propagation in ice and water in `data/mmc/(ice|water)_(allm97|bb).npz` for two different cross section parameterizations. The detector response probability must first be defined in `scripts/mu/pl.py` as a function of the muon energy **at the detector**. Then, pass the function name to the `--plight` argument and construct the overall muon reaching and detection probability with the following command, for example.

```bash
cd scripts/mu
./mu.py ice_allm97 -o mymudet.npz --plight pl_step_1000
```

To use the newly generated file, pass the stem without file extension as a string to the `prpl` argument.
```python
passing(enu, cos_theta, prpl='mymudet')`.
```

For many different characterisations of the detector response (e.g. for different depths or different directions) this process can be inconvenient. In this case one can also directly use a function for pl:
```python
from nuVeto.mu import interp

pl=lambda emu: #some function of muon energy

prpl=interp("ice_allm97",pl)
enu=1e3
cos_theta=0.5
pf = passing(enu, cos_theta, prpl=prpl)
```

## Contributers
_Carlos Arguelles, Sergio Palomares-Ruiz, Austin Schneider, Logan Wille, Tianlu Yuan_
