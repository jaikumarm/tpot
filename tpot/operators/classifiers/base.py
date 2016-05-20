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

class Classifier(Operator):
    """Parent class for classifiers for TPOT"""
    def __init__(self, import_hash):
        super(self.__class__, self).__init__(import_hash)

    def __call__(self, input_df, *args, **kwargs):
        classifier, classifier_kwargs = self.operator_code(input_df, *args, **kwargs)

        return self._train_model_and_predict(input_df, classifier, **classifier_kwargs)

    def _train_model_and_predict(self, input_df, model, **kwargs):
        """Fits an arbitrary sklearn classifier model with a set of keyword parameters

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-neares
        model: sklearn classifier
            Input model to fit and predict on input_df
        kwargs: unpacked parameters
            Input parameters to pass to the model's constructor, does not need
            to be a dictionary

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated
            according to the classifier's predictions. Also adds the
            classifiers's predictions as a 'SyntheticFeature' column.

        """
        input_df = input_df.copy()

        training_features = input_df.\
            loc[input_df['group'] == 'training'].\
            drop(self.non_feature_columns, axis=1).values
        training_classes = input_df.\
            loc[input_df['group'] == 'training', 'class'].values

        # If there are no features left (i.e., only 'class', 'group', and
        # 'guess' remain in the DF), then there is nothing to do
        if len(training_features.columns) == 0:
            return input_df

        # Try to seed the random_state parameter if the model accepts it.
        try:
            random_state = self.default_seed

            clf = model(random_state=random_state, **kwargs)
            clf.fit(training_features, training_classes)
        except TypeError:
            clf = model(**kwargs)
            clf.fit(training_features, training_classes)

        all_features = input_df.drop(self.non_feature_columns, axis=1).values
        input_df.loc[:, 'guess'] = clf.predict(all_features)

        # Also store the guesses as a synthetic feature
        sf_hash = '-'.join(sorted(input_df.columns.values))
        # Use the classifier object's class name in the synthetic feature
        sf_hash += '{}'.format(clf.__class__)
        sf_hash += '-'.join(kwargs)
        sf_identifier = 'SyntheticFeature-{}'.format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())
        input_df.loc[:, sf_identifier] = input_df['guess'].values

        return input_df
