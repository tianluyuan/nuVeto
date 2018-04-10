#!/usr/bin/env python

from distutils.core import setup

setup(
    name='nuVeto',
    version='1.0.0',
    author='C. Arguelles, S. Palomares-Ruiz, A. Schneider, L. Wille, and T. Yuan',
    author_email='caad@mit.edu, Sergio.Palomares.Ruiz@ific.uv.es, aschneider@icecube.wisc.edu, lwille@icecue.wisc.edu, and tyuan@icecube.wisc.edu',
    description='Package implements the formalism for calculating passing fraction as discussed in arXiv:XXXX.XXXX.',
    long_description=open('README.md').read(),
    url='https://github.com/arguelles/SelfVeto.git',
    packages=['nuVeto','nuVeto.external'],
    package_dir={'nuVeto': 'nuVeto'},
    package_data={'nuVeto':["data/decay_distributions/*.npz",'data/prpl/*.pkl','data/corsika/*.pkl']},
    #data_files={'nuVeto':["data/decay_distributions/*.npz",'data/prpl/*.pkl','data/corsika/*.pkl']},
    requires=['numpy', 'scipy', 'functools32', 'matplotlib', 'MCeq']
    )
