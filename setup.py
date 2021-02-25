from setuptools import setup, find_packages


setup(
    name='nuVeto',
    version='2.1.4',
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
    install_requires=['numpy', 'scipy', 'MCEq[MKL]'],
    extras_require={
        'plotting':  ['matplotlib'],
        'resources':  ['pythia8', 'matplotlib', 'argparse', 'pandas'],
        'testing': ['pytest==4.6.4']
    },
    python_requires='>=3.3',
    license=open('LICENSE').readline().split()[0],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        ],
    )
