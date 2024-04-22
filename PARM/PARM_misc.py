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

from .version import __version__

def log(message):
    v = 'PARM v' + __version__
    print(f"[{v}] {message}", flush=True)