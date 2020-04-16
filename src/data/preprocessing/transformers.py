import datetime
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RollingAverageNanTransformer(TransformerMixin):
    """
    Replace NaN values with rolling average.

    Returned dataframe contains only transformed column.
    """
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
        empty_rows['date'] = pd.to_datetime(empty_rows.timestamp)
        for i, row in empty_rows.iterrows():
            site_id = row['site_id']
            timestamp = row['timestamp']
            date = row['date']
            prew_timestamp = str(
                date - datetime.timedelta(hours=self.window_size)
            )

            df_site = df[df.site_id == site_id]
            df_slice = df_site[df_site.timestamp >= prew_timestamp]
            df_slice = df_slice[df_slice.timestamp < timestamp]
            df_slice = df_slice[self.column].dropna().values

            if len(df_slice) > 0:
                fill_in_value = np.mean(df_slice).round(1)
            else:
                current_hour = date.hour
                fill_in_value = self \
                    .averages_per_site_per_hour[site_id][current_hour]

            # Fill in mean value now in case next value is also NaN
            df.loc[df.index == i, self.column] = fill_in_value
        return df[[self.column]]

    def get_feature_names(self):
        return self.column


class OutlierTransformer(TransformerMixin):
    """
    Replace outliers with 5th percentile or 95th percentile.

    Entire dataframe is returned.
    """
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

    Returned dataframe contains only transformed column.
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

    Returned dataframe contains only transformed column.
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

    Entire dataframe is returned.
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
            raise ValueError(
                'Either `threshold` or `specific_value` parameter must '
                'have some value.'
            )
        return df[rows_filter]

    def get_feature_names(self):
        return [self.feature]


class OneHotEncoderTransformer(TransformerMixin):
    """
    Encode given `column` using one hot encoding.

    Returned dataframe contains only transformed columns.
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

    Returned dataframe contains only transformed column if `all_columns`
    is False else it returns entire dataframe with scaled `column`.
    """
    def __init__(self, column, new_column=None, all_columns=False):
        """
        Create a new instance of `StandardScalerTransformer` class.

        :param column: str, column to be scaled.
        :param new_column: str (default: None), column where scaled
            values are to be saved. If `None`, scaled values are saved
            to `column`.
        """
        self.column = column
        self.new_column = new_column
        self.all_columns = all_columns
        self.columns = None

    def fit(self, df, y=None, **fit_params):
        if self.new_column is None:
            self.columns = [self.column]
        else:
            self.columns = [self.column, self.new_column]

        if self.all_columns:
            self.columns = df.columns.values.tolist()
            if self.new_column is not None:
                self.columns.append(self.new_column)

        self.scaler = StandardScaler()
        self.scaler.fit(df[[self.column]])
        return self

    def transform(self, df, **transform_params):
        if self.all_columns:
            column = self.column
            if self.new_column is not None:
                column = self.new_column
            df[column] = self.scaler.transform(df[[self.column]])
        else:
            if self.new_column is None:
                data = self.scaler.transform(df[[self.column]])
            else:
                data = {
                    self.column: df[self.column],
                    self.new_column: self.scaler.transform(df[[self.column]])
                }
            df = pd.DataFrame(data, columns=self.columns)

        return df

    def get_feature_names(self):
        return self.columns


class AddPreviousMeterReadingTransformer(TransformerMixin):
    """
    Adds meter reading from previous n-hours as new features.

    Returns entire dataframe with `time_horizon` new features
    representing previous meter readings.
    """
    def __init__(self, time_horizon, sample_length=3600):
        """
        Sets `time_horizon` and `sample_length`.

        :param time_horizon: int, number of previous values that will be
            added as new features
        :param sample_length: int (default: 3600), number of seconds
            between two samples
        """
        self.time_horizon = time_horizon
        self.sample_length = sample_length
        self.columns = ['meter_reading']
        for i in range(1, self.time_horizon + 1):
            self.columns.append(f'meter_reading_scaled_{i}')
        self.min_value = None

    def _get_new_date(self, row, step=1):
        """
        Gets new date `step` times `self.sample_length` in the future
        compared to date of `row`.

        :param row: pd.Series, row for which future date will be found
        :param step: int, how many steps to look into future
        :return: Datetime, future date
        """
        new_date = row['date'] + datetime.timedelta(
            seconds=self.sample_length * step
        )
        return [row['building_id'], row['meter'], str(new_date)]

    def fit(self, df, y=None, **fit_params):
        self.min_value = df['meter_reading_scaled'].min()
        return self

    def transform(self, df, **transform_params):
        df = df.copy().reset_index(drop=True)
        df['date'] = pd.to_datetime(df['timestamp'])
        data = {}
        for i in range(1, self.time_horizon + 1):
            column = f'timestamp_{i}'
            aux = df.apply(lambda x: self._get_new_date(x, step=i), axis=1)
            data['building_id'] = aux.apply(lambda x: x[0])
            data['meter'] = aux.apply(lambda x: x[1])
            data[column] = aux.apply(lambda x: x[2])
        data['meter_reading_scaled'] = df['meter_reading_scaled']

        new_df = pd.DataFrame(data)
        for i in range(1, self.time_horizon + 1):
            timestamp = f'timestamp_{i}'
            aux_df = new_df[[
                'building_id',
                'meter',
                'meter_reading_scaled',
                timestamp
            ]]
            aux_df.columns = [
                'building_id',
                'meter',
                'meter_reading_scaled',
                'timestamp'
            ]

            # Dataframes should be joined by unique values
            df = df.join(
                aux_df.set_index([
                    'building_id',
                    'meter',
                    'timestamp'
                ]),
                on=['building_id', 'meter', 'timestamp'],
                rsuffix=f"_{i}"
            ).drop_duplicates()
        df = df.fillna(self.min_value)
        return df[self.columns]

    def get_feature_names(self):
        return self.columns
