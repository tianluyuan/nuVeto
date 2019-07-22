from setuptools import setup, find_packages

setup(
    name='nuVeto',
    version='1.0.0',
    author='C. Arguelles, S. Palomares-Ruiz, A. Schneider, L. Wille, and T. Yuan',
    author_email='caad@mit.edu, Sergio.Palomares.Ruiz@ific.uv.es, aschneider@icecube.wisc.edu, lwille@icecue.wisc.edu, and tyuan@icecube.wisc.edu',
    description='Package implements the formalism for calculating passing fraction as discussed in arXiv:XXXX.XXXX.',
    long_description=open('README.md').read(),
    url='https://github.com/tianluyuan/nuVeto.git',
    packages=find_packages('./'),
    package_data={
        'nuVeto':['data/decay_distributions/*.npz','data/prpl/*.pkl','data/corsika/*.pkl'],
        'nuVeto.resources.mu':['mmc/ice*.pklz']
    },
    install_requires=['numpy',
                      'scipy',
                      'functools32'],
    extras_require={
        'plotting':  ['matplotlib', 'pandas'],
        'resources':  ['pythia8', 'matplotlib', 'argparse', 'pandas'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
