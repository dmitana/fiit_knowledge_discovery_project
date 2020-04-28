import datetime
from functools import reduce
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
        aux.drop_duplicates(subset=['site_id', 'timestamp'], inplace=True)

        for i in aux.site_id.unique():
            self.averages_per_site_per_hour[i] = aux[aux.site_id == i] \
                .groupby(by="timestamp_hour").mean()[self.column].values
        return self

    def transform(self, df, **transform_params):
        if df[self.column].isna().sum() == 0:
            return df

        df = df.copy()
        df['indices'] = df.index

        group_by_empty_rows = df[df[self.column].isna()] \
            .groupby(by=['site_id', 'timestamp']).agg({'indices': list})
        group_by_empty_rows['site_id'] = group_by_empty_rows.index \
            .get_level_values('site_id')
        group_by_empty_rows['timestamp'] = group_by_empty_rows.index \
            .get_level_values('timestamp')
        group_by_empty_rows['date'] = pd.to_datetime(
            group_by_empty_rows.index.get_level_values('timestamp')
        )
        weather_df = df.drop_duplicates(subset=['site_id', 'timestamp'])
        for _, row in group_by_empty_rows.iterrows():
            site_id = row['site_id']
            timestamp = row['timestamp']
            date = row['date']

            prew_timestamp = str(
                date - datetime.timedelta(hours=self.window_size)
            )

            df_site = weather_df[weather_df.site_id == site_id]
            df_slice = df_site[df_site.timestamp >= prew_timestamp]
            df_slice = df_slice[df_slice.timestamp < timestamp]
            df_slice = df_slice[self.column].dropna().values

            if len(df_slice) > 0:
                fill_in_value = np.mean(df_slice).round(1)
            else:
                current_hour = date.hour
                if site_id in self.averages_per_site_per_hour:
                    fill_in_value = self \
                        .averages_per_site_per_hour[site_id][current_hour]
                else:
                    fill_in_value = reduce(
                        lambda x, y: x + y[current_hour],
                        self.averages_per_site_per_hour.values(),
                        0
                    ) / len(self.averages_per_site_per_hour)

            # Fill in mean value now in case next value is also NaN
            for index in row['indices']:
                df.loc[df.index == index, self.column] = fill_in_value
                weather_df.loc[weather_df.index == index, self.column] = \
                    fill_in_value
        return df

    def get_feature_names(self):
        return self.column


class OutlierTransformer(TransformerMixin):
    """
    Replace outliers with 5th percentile or 95th percentile.

    Entire dataframe is returned.
    """

    def __init__(self, column, group_by_columns=None):
        self.column = column
        self.group_by_columns = group_by_columns

    def fit(self, df, y=None, **fit_params):
        if self.group_by_columns is not None:
            df = df.drop_duplicates(subset=self.group_by_columns)

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
        self.one_hot_encoder = OneHotEncoder(
            sparse=False, handle_unknown='ignore'
        )
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

    def __init__(
        self,
        column,
        new_column=None,
        group_by_column=None,
        all_columns=False
    ):
        """
        Create a new instance of `StandardScalerTransformer` class.

        :param column: str, column to be scaled.
        :param new_column: str (default: None), column where scaled
            values are to be saved. If `None`, scaled values are saved
            to `column`.
        :param group_by_column: str|list (default: None), column to be
            data grouped by.
        :param all_columns: bool (default: False), whether to return
            entire dataframe.
        """
        self.column = column
        self.new_column = new_column
        self.group_by_column = group_by_column
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

        if self.group_by_column is not None:
            values = df.groupby(by=self.group_by_column).min()
            values = values[self.column].values.reshape(-1, 1)
        else:
            values = df[[self.column]]

        self.scaler = StandardScaler()
        self.scaler.fit(values)
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
        Set `time_horizon` and `sample_length`.

        :param time_horizon: int, number of previous values that will be
            added as new features
        :param sample_length: int (default: 3600), number of seconds
            between two samples
        """
        self.time_horizon = time_horizon
        self.sample_length = sample_length
        self.columns = []
        for i in range(1, self.time_horizon + 1):
            self.columns.append(f'meter_reading_scaled_{i}')
        self.min_value = None

    def fit(self, df, y=None, **fit_params):
        self.min_value = df['meter_reading_scaled'].min()
        return self

    def transform(self, df, **transform_params):
        df = df.copy().reset_index(drop=True)

        timestamps = pd.to_datetime(df.timestamp.unique())

        timestamps_aux = []
        for i, timestamp in enumerate(timestamps):
            timestamps_aux.append([str(timestamp)])
            for j in range(1, self.time_horizon + 1):
                new_timestamp = timestamp + datetime.timedelta(hours=j)
                timestamps_aux[i].append(str(new_timestamp))

        columns = ['timestamp']
        columns.extend([
            f'timestamp_{i}' for i in range(1, self.time_horizon + 1)
        ])
        df_timestamp = pd.DataFrame(timestamps_aux, columns=columns)

        new_df = df.join(
            df_timestamp.set_index(['timestamp']),
            on='timestamp'
        )

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
