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
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

from .base import Preprocessor
from sklearn.decomposition import RandomizedPCA

class PCA(Preprocessor):
    """Uses scikit-learn's RandomizedPCA to transform the feature set

    Parameters
    ----------
    n_components: int
        The number of components to keep
    iterated_power: int
        Number of iterations for the power method. [1, 10]

    """
    import_hash = {'sklearn.decomposition': ['RandomizedPCA']}
    sklearn_class = RandomizedPCA

    def __init__(self):
        pass


    def preprocess_args(self, n_components: int, iterated_power: int):
        if n_components < 1:
            n_components = 1
        else:
            n_components = min(n_components, len(self.training_features))

        # Thresholding iterated_power [1, 10]
        iterated_power = min(10, max(1, iterated_power))

        return {
            'n_components': n_components,
            'iterated_power': iterated_power,
            'copy': False
        }
