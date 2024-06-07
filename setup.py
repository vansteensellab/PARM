#!/usr/bin/env python3

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

__version__ = '0.0.0'
exec(open('PARM/version.py').read())


setup(name='parm',
    version=__version__,
    description='PARM: Promoter Activity Regulatory Model',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/vansteensellab/PARM',
    author='PARM developers',
    license='GPLv3',
    packages=['PARM'],
    entry_points={"console_scripts": ['parm = PARM.__main__:main']},
    python_requires='>=3.9',
    install_requires=['torchsummary']
)