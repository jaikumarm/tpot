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

from tpot.operators import Operator

class Preprocessor(Operator):
    """Parent class for Preprocessors for TPOT"""
    def __init__(self, import_hash):
        super(self.__class__, self).__init__(import_hash)

    def __call__(input_df, *args, **kwargs):
        training_features = input_df.loc[input_df['group'] == 'training'].\
            drop(self.non_feature_columns, axis=1)

        # Return input_df if there are no feature columns
        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # Run the feature-preprocessor
        modified_df = self.operator_code(training_features, *args, **kwargs)

        # Add non_feature_columns back to DataFrame
        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        # Translate non-string column titles into strings
        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df
