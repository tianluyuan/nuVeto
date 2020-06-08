from setuptools import setup, find_packages

setup(
    name='nuVeto',
    version='2.1',
    maintainer='Tianlu Yuan',
    maintainer_email='tyuan@icecube.wisc.edu',
    description='Package implements the formalism for calculating passing fraction as discussed in JCAP07(2018)047.',
    long_description='This package calculates the effect of a detector veto on the high-energy atmospheric neutrino flux via detection of muons that reach the detector. The result calculated is the passing-flux or passing-fraction of atmospheric neutrinos as a function of energy and zenith angle.',
    url='https://github.com/tianluyuan/nuVeto.git',
    packages=find_packages('./'),
    package_data={
        'nuVeto':['data/decay_distributions/*.npz','data/prpl/*.pkl','data/corsika/*.pkl'],
        'nuVeto.resources.mu':['mmc/ice*.pklz']
    },
    install_requires=['scipy',
                      'numpy',
                      'MCEq[MKL]'],
    extras_require={
        'plotting':  ['matplotlib'],
        'resources':  ['pythia8', 'matplotlib', 'argparse', 'pandas']
    },
    setup_requires=['pytest-runner'],
    python_requires='>=3.3',
    tests_require=['pytest==4.6.4'],
    )
