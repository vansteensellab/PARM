#!/usr/bin/env python3
"""
Copyright 2024 PARM developers
https://github.com/vansteensellab/PARM

This file is part of PARM. PARM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. PARM is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with PARM.
If not, see <http://www.gnu.org/licenses/>.
"""
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

__version__ = '0.0.0'
exec(open('PARM/version.py').read())


setup(name='PARM',
      version=__version__,
      description='PARM: Promoter Activity Regulatory Model',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/vansteensellab/PARM',
      author='PARM developers',
      license='GPLv3',
      packages=['PARM'],
      entry_points={"console_scripts": ['PARM = PARM.__main__:main']},
      python_requires='>=3.9')