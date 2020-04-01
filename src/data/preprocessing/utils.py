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


def preprocess_and_merge_data(
    buildings_data,
    weather_data,
    meter_data,
    buildings_fu,
    weather_fu,
    meter_fu,
    **merge_data_kwargs
):
    """
    Preprocess buildings data, weather data and meter data using
    corresponding feature unions and merge them into one data set.

    :param buildings_data: pandas.DataFrame, buildings data.
    :param weather_data: pandas.DataFrame, weather data.
    :param meter_data: pandas.DataFrame, meter data.
    :param buildings_fu: CustomFeatureUnion, buildings data
        preprocessing.
    :param weather_fu: CustomFeatureUnion, weather data preprocessing.
    :param meter_fu: CustomFeatureUnion, meter data preprocessing.
    :param merge_data_kwargs: dict, keyword arguments of `merge_data`
        function.
    :return:
        pandas.DataFrame, dataframe containing only target value
            (meter_reading).
        pandas.DataFrame, dataframe containing other columns.
    """
    # Preprocess data using corresponding feature unions
    buildings_features = buildings_fu.union_features(buildings_data)
    weather_features = weather_fu.union_features(weather_data)
    meter_features = meter_fu.union_features(meter_data)

    # Merge data and drop useless columns
    data = merge_data(
        buildings_features,
        weather_features,
        meter_features,
        **merge_data_kwargs
    )
    data.drop(columns=['building_id', 'timestamp', 'site_id'], inplace=True)

    return data[['meter_reading']], \
        data[data.columns.difference(['meter_reading'])]
