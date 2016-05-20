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

import pandas as pd
from inspect import signature

class Operator(object):
    """Base class for operators in TPOT"""
    def __init__(self, import_hash):
        self.import_hash = import_hash
        self.default_seed = 42 # Default value for random_state when appropriate
        self.non_feature_columns = ['class', 'group', 'guess']

    def export(self, *args, **kwargs):
        pass

    def parameter_types(self):
        arg_types = [pd.DataFrame] # First argument is always a DataFrame

        # Inspect operator_code function to get parameter information
        # Uses function parameter annotations to determine parameter types
        operator_parameters = signature(self.operator_code).parameters
        param_names = [key for key in operator_parameters.keys()][1:] # Skip input_df

        for param in param_names:
            annotation = operator_parameters[param].annotation

            # Raise RuntimeError if a type is not annotated
            if annotation is signature.empty:
                raise RuntimeError('Undocumented argument type for {} in operator {}'.\
                    format(param, self.operator_code.__self__.__class__))
            else:
                arg_types.append(annotation)

        return (arg_types, pd.DataFrame) # Return type is always a DataFrame

    @classmethod
    def inheritors(cls):
        """Returns set of all operators defined"""
        operators = set()

        # Search two levels deep and report leaves in inheritance tree
        for child in cls.__subclasses__():
            for grandchild in child.__subclasses__():
                operators.add(grandchild)

        return operators
