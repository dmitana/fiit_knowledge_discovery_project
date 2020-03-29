import datetime
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RollingAverageNanTransformer(TransformerMixin):
    """Replace NaN values with rolling average."""
    def __init__(self, column, window_size=5):
        self.column = column
        self.window_size = window_size

    def fit(self, df, y=None, **fit_params):
        self.averages_per_site_per_hour = {}
        aux = df.copy()
        aux['timestamp_hour'] = \
            pd.to_datetime(aux['timestamp']).apply(lambda x: x.hour)
        for i in range(16):
            self.averages_per_site_per_hour[i] = aux[aux.site_id == i] \
                .groupby(by="timestamp_hour").mean()[self.column].values
        return self

    def transform(self, df, **transform_params):
        df = df.copy()
        empty_rows = df[df[self.column].isna()][['site_id', 'timestamp']]
        for site_id, timestamp in empty_rows.values:
            # Obtain timestamp self.window_size hours before timestamp
            current_timestamp = datetime.datetime \
                .strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            prew_timestamp = current_timestamp - \
                datetime.timedelta(hours=self.window_size)
            prew_timestamp = datetime.datetime \
                .strftime(prew_timestamp, "%Y-%m-%d %H:%M:%S")

            df_slice = df[
                (df.site_id == site_id) &
                (df.timestamp >= prew_timestamp) &
                (df.timestamp < timestamp)
            ][self.column].dropna().values
            if len(df_slice) > 0:
                fill_in_value = np.mean(df_slice).round(1)
            else:
                current_hour = current_timestamp.hour
                fill_in_value = self \
                    .averages_per_site_per_hour[site_id][current_hour]

            # Fill in mean value now in case next value is also NaN
            df.loc[
                (df.site_id == site_id) &
                (df.timestamp == timestamp),
                self.column
            ] = fill_in_value
        return df


class OutlierTransformer(TransformerMixin):
    """Replace outliers with 5th percentile or 95th percentile."""
    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        iqr = df[self.column].quantile(q=0.75) - \
            df[self.column].quantile(q=0.25)

        self.upper_bound = df[self.column].quantile(q=0.75) + 1.5 * iqr
        self.lower_bound = df[self.column].quantile(q=0.25) - 1.5 * iqr
        self.percentile_5 = df[self.column].quantile(q=0.05)
        self.percentile_95 = df[self.column].quantile(q=0.95)
        return self

    def transform(self, df, **transform_params):
        df = df.copy()
        df.loc[df[self.column] > self.upper_bound, self.column] = \
            self.percentile_95
        df.loc[df[self.column] < self.lower_bound, self.column] = \
            self.percentile_5
        return df


class PrimaryUseTransformer(TransformerMixin):
    """
    Merge less numerous categories of `primary_use` feature to
    category `Other`.

    Returned is dataframe containing only transformed columns.
    Transformation is done in place.
    """
    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        return self.merge_categories(df)[['primary_use']]

    def merge_categories(self, df):
        df.loc[
            (
                ~df['primary_use'].str.contains(
                    'Lodging/residential',
                    regex=True
                )
            ) &
            (
                ~df['primary_use'].str.contains(
                    'Public services',
                    regex=True
                )
            ) &
            (
                ~df['primary_use'].str.contains(
                    'Entertainment/public assembly',
                    regex=True
                )
            ) &
            (
                ~df['primary_use'].str.contains(
                    'Office',
                    regex=True
                )
            ) &
            (
                ~df['primary_use'].str.contains(
                    'Education',
                    regex=True
                )
            ),
            'primary_use'
        ] = 'Other'
        return df

    def get_feature_names(self):
        return ['primary_use']


class FeatureSelectorTransformer(TransformerMixin):
    """
    Select given `feature` from given `df`.

    Returned is dataframe containing only transformed columns.
    """
    def __init__(self, feature):
        self.feature = feature

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df[[self.feature]]

    def get_feature_names(self):
        return [self.feature]


class ValuePicker(TransformerMixin):
    """
    Selects only those rows from given `feature` of dataframe `df`,
    where value fulfills either `threshold` or `specific_value`
    constrains.
    """
    def __init__(self, feature, threshold=None, specific_value=None):
        """
        :param feature: str, name of feature for which values will be
            picked
        :param threshold: int, only rows where `feature` values are
            smaller than threshold will be returned
        :param specific_value: int, only rows where `feature` values
            are equal to specific_value will be returned
        """
        self.feature = feature
        self.threshold = threshold
        self.specific_value = specific_value

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if self.threshold is not None:
            rows_filter = df[self.feature] < self.threshold
        elif self.specific_value is not None:
            rows_filter = df[self.feature] == self.specific_value
        else:
            raise ValueError("Either `threshold` or `specific_value` parameter"
                             " must have some value.")
        return df[rows_filter]


class OneHotEncoderTransformer(TransformerMixin):
    """
    Encode given `column` using one hot encoding.

    Returned is dataframe containing only transformed columns.
    """
    def __init__(self, column):
        """
        Create a new instance of `OneHotEncoderTransformer` class.

        :param column: str, column to be one hot encoded.
        """
        self.column = column

    def fit(self, df, y=None, **fit_params):
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.one_hot_encoder.fit(df[[self.column]])
        self.categories = self.one_hot_encoder.categories_[0]
        return self

    def transform(self, df, **transform_params):
        return pd.DataFrame(
            self.one_hot_encoder.transform(df[[self.column]]),
            columns=self.categories
        )

    def get_feature_names(self):
        return self.categories


class StandardScalerTransformer(TransformerMixin):
    """
    Scale given `column` using z-score normalization.

    Returned is dataframe containing only transformed columns.
    """
    def __init__(self, column):
        """
        Create a new instance of `StandardScalerTransformer` class.

        :param column: str, column to be scaled.
        """
        self.column = column

    def fit(self, df, y=None, **fit_params):
        self.scaler = StandardScaler()
        self.scaler.fit(df[[self.column]])
        return self

    def transform(self, df, **transform_params):
        return pd.DataFrame(
            self.scaler.transform(df[[self.column]]),
            columns=[self.column]
        )

    def get_feature_names(self):
        return [self.column]
