import pandas as pd
import datetime


def merge_data(building, weather, meter, hour_time_diff=1):
    """
    Merges building data, weather data and meter data to one data set.

    Before merge, weather data are shifted by hour_time_diff. This is
    done because we need past data to predict future values of meter
    reading.

    :param building: pandas.DataFrame, building data.
    :param weather: pandas.DataFrame, weather data.
    :param meter: pandas.DataFrame, meter data.
    :param hour_time_diff: int (default: 1), number of hours the weather
        data will be shifted by.
    :return: pandas.DataFrame, merged data set.
    """
    weather_aux = weather.copy()
    weather_aux['timestamp'] = weather_aux['timestamp'].apply(
        lambda x: (
            pd.to_datetime(x) +
            datetime.timedelta(hours=hour_time_diff)
        ).strftime("%Y-%m-%d %H:%M:%S")
    )
    meter_building = pd.merge(meter, building, on='building_id')
    return pd.merge(
        meter_building,
        weather_aux,
        how='left',
        on=['site_id', 'timestamp']
    )[1:]


def get_outliers(df, column):
    """
    Gets outliers of given `column` in given `df`.

    :param df: pandas.DataFrame, dataframe.
    :param column: str, dataframe's column to get outliers of.
    :return: pandas.DataFrame, dataframe containing outliers.
    """
    iqr = df[column].quantile(q=0.75) - df[column].quantile(q=0.25)
    upper_bound = df[column].quantile(q=0.75) + 1.5 * iqr
    lower_bound = df[column].quantile(q=0.25) - 1.5 * iqr
    print(f'lower bound: {lower_bound}\nupper bound: {upper_bound}')
    return df[(df[column] > upper_bound) | (df[column] < lower_bound)]
