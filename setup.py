# python3

"""Install script for setuptools."""

import datetime
from importlib import util as import_util
import sys

from setuptools import find_packages
from setuptools import setup

reverb_requirements = [
    'dm-reverb-nightly==0.1.0.dev20200708',
    'tf-nightly==2.4.0.dev20200708',
]

tf_requirements = [
    'tf-nightly==2.4.0.dev20200708',
    'tfp-nightly==0.12.0.dev20200717',
    'dm-sonnet',
    'trfl',
]

jax_requirements = [
    'jax',
    'jaxlib',
    'dm-haiku',
    'optax',
    'rlax @ git+git://github.com/deepmind/rlax.git#egg=rlax',
    'dataclasses',  # Back-port for Python 3.6.
]

env_requirements = [
    'bsuite @ git+git://github.com/deepmind/bsuite.git#egg=bsuite',
    'dm-control',
    'gym',
    'gym[atari]',
]

testing_requirements = [
    'pytype',
    'pytest-xdist',
]

# Use the first paragraph of our README as the long_description.
with open('README.md', 'r') as fh:
  long_description = fh.read().split('\n\n')[4]

# Add a link to github.
long_description += '\n\nFor more information see our '
long_description += '[github repository](https://github.com/RaoulDrake/ftw).'

# Get the version from metadata.
version = '0.1.8'

setup(
    name='ftw',
    version=version,
    description='A Python library implementing the For The Win (FTW) agent as introduced by Deepmind.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Johannes Tochtermann',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'dm-tree',
        'numpy<1.19.0, >=1.16.0',
        'pillow',
        'dm-acme==0.1.8'
    ] + reverb_requirements + tf_requirements + jax_requirements + env_requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)