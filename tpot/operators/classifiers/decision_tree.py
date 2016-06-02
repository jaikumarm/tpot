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

from .base import Classifier
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Classifier):
    """Fits a decision tree classifier

    Parameters
    ----------
    max_features: int
        Number of features used to fit the decision tree; must be a positive value
    max_depth: int
        Maximum depth of the decision tree; must be a positive value

    """
    import_hash = {'sklearn.tree': ['DecisionTreeClassifier']}
    sklearn_class = DecisionTreeClassifier

    def __init__(self):
        super(self.__class__, self).__init__()

    def preprocess_args(max_features: int, max_depth: int):
        if max_features <= 1:
            max_features = 'auto'
        elif max_features > len(self.training_features.columns)
            max_features = len(self.training_features.columns)

        max_depth = max(max_depth, 1)

        return {
            'max_features': max_features,
            'max_depth': max_depth
        }
