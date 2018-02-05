import sys
from os import path,listdir
from setuptools import setup,find_packages
__version__ = "0.0.4.7"
REQUIRES = [
    'pandas>=0.20.1',
    'numpy>=1.12.1',
    'scipy>=0.19.1'
]
setup(
    name = 'GMV-MVE trading algorithm',
    author = 'zhhrozhh',
    author_email = 'zhangh40@msu.edu',
    url = 'https://github.com/zhhrozhh/GMV-MVE-trading-algorithm',
    version = __version__,
    license = 'MIT',
    classifiers = [
        'Programming Language :: Python :: 3.5'
    ],
    keywords = 'algorithmic trading',
    packages = find_packages(),
    install_requires = REQUIRES
)
