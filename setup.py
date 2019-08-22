from setuptools import setup, find_packages

setup(
    name='nuVeto',
    version='2.0',
    author='C. Arguelles, S. Palomares-Ruiz, A. Schneider, L. Wille, and T. Yuan',
    author_email='caad@mit.edu, Sergio.Palomares.Ruiz@ific.uv.es, aschneider@icecube.wisc.edu, lwille@icecube.wisc.edu, and tyuan@icecube.wisc.edu',
    description='Package implements the formalism for calculating passing fraction as discussed in JCAP07(2018)047.',
    long_description=open('README.md').read(),
    url='https://github.com/tianluyuan/nuVeto.git',
    packages=find_packages('./'),
    package_data={
        'nuVeto':['data/decay_distributions/*.npz','data/prpl/*.pkl','data/corsika/*.pkl'],
        'nuVeto.resources.mu':['mmc/ice*.pklz']
    },
    install_requires=['functools32',
                      'scipy<1.3.0',
                      'numpy<1.17.0',
                      'MCEq[MKL]'],
    extras_require={
        'plotting':  ['matplotlib'],
        'resources':  ['pythia8', 'matplotlib', 'argparse', 'pandas']
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.6.4']
    )
