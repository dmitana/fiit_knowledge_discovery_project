import datetime
import numpy as np

from sklearn.base import TransformerMixin


class RollingAverageNanFiller(TransformerMixin):
    def __init__(self, column, window_size=5):
        self.column = column
        self.window_size = window_size

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df = df.copy()
        empty_rows = df[df[self.column].isna()][['site_id', 'timestamp']]
        for site_id, timestamp in empty_rows.values:
            # obtain timestamp self.window_size hours before timestamp
            prew_timestamp = datetime.datetime\
                .strptime(timestamp, "%Y-%m-%d %H:%M:%S") - \
                datetime.timedelta(hours=self.window_size)
            prew_timestamp = datetime.datetime\
                .strftime(prew_timestamp, "%Y-%m-%d %H:%M:%S")

            fill_in_value = np.mean(
                df[(df.site_id == site_id) &
                   (df.timestamp >= prew_timestamp) &
                   (df.timestamp < timestamp)][self.column].values
            ).round(1)

            # fill in mean value now in case next value is also NaN
            df.loc[(df.site_id == site_id) &
                   (df.timestamp == timestamp), self.column] = fill_in_value
        return df


class OutlierTransformer(TransformerMixin):
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
