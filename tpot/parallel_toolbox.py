# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

from copy import deepcopy
from multiprocessing import Pool

from deap import base


class ParallelToolbox(base.Toolbox):
    """Runs DEAP genetic algorithms over multiple cores"""

    def __init__(self):
        super(base.Toolbox, self).__init__()

        self.register('clone', deepcopy)

        # Replace default mapping function with parallelized version
        pool = Pool()
        self.register('map', pool.map)

    def __getstate__(self):
        self_dict = self.__dict__.copy()

        # Delete map function to prevent the Pool from pickling itself
        del self_dict['map']

        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
