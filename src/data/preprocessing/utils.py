import datetime
import pandas as pd
import numpy as np


def split_meter_data(
    meter_data,
    buildings_data,
    train_size=1.0,
    dev_size=0.0,
    test_size=0.0,
    seed=13
):
    """
    Split meter data to train, dev and test sets.

    Created sets are disjoint with respect to containing buildings and
    each split contains buildings from each site.

    :param meter_data: pandas.DataFrame, meter data.
    :param buildings_data: pandas.DataFrame, buildings data.
    :param train_size: float (default: 1.0), size of train set.
    :param dev_size: float (default: 0.0), size of dev set.
    :param test_size: float (default: 0.0), size of test set.
    :param seed: int (default: 13), random seed.
    :return:
        pandas.DataFrame, train meter data.
        pandas.DataFrame, dev meter data.
        pandas.DataFrame, test meter data.
    """
    n_sites = len(buildings_data.site_id.unique())
    sites = [[] for _ in range(n_sites)]

    # Get all building for particular site
    for _, row in buildings_data.iterrows():
        sites[row['site_id']].append(row['building_id'])

    # Get train, dev and test buildings from each site
    train_buildings, dev_buildings, test_buildings = [], [], []
    for site in sites:
        # Compute indices
        n_buildings = len(site)
        idx1 = int(n_buildings * 0.1)
        idx2 = idx1 * 2

        # Shuffle
        np.random.seed(seed)
        perm = np.random.permutation(n_buildings)
        buildings = np.array(site)[perm]

        # Split to dev, test and train
        dev_buildings_aux, test_buildings_aux, train_buildings_aux = \
            np.split(buildings, [idx1, idx2])

        # Add buildings from current site to resulting lists
        train_buildings.extend(train_buildings_aux)
        dev_buildings.extend(dev_buildings_aux)
        test_buildings.extend(test_buildings_aux)

    # Split meter data to train, dev and test
    train_meter_data = meter_data[meter_data.building_id.isin(train_buildings)]
    dev_meter_data = meter_data[meter_data.building_id.isin(dev_buildings)]
    test_meter_data = meter_data[meter_data.building_id.isin(test_buildings)]

    return train_meter_data, dev_meter_data, test_meter_data


def filter_data(
    buildings_data,
    weather_data,
    meter_data,
    site_id=None,
    primary_use=None,
    meter=None,
    meter_reading=None
):
    """
    Filter data before preprocessing.

    :param buildings_data: pandas.DataFrame, buildings data.
    :param weather_data: pandas.DataFrame, weather data.
    :param meter_data: pandas.DataFrame, meter data.
    :param site_id: int|list (default: None), site id to be used.
    :param primary_use: int|list (default: None), primary use to be used.
    :param meter: int|list (default: None), meter type to be used.
    :param meter_reading: float (default: None), upper bound of meter
        reading to be used.
    :return:
        pandas.DataFrame, dataframe containing filtered buildings data.
        pandas.DataFrame, dataframe containing filtered weather data.
        pandas.DataFrame, dataframe containing filtered meter data.
    """
    if site_id is not None:
        site_id = site_id if type(site_id) in (list, tuple) else [site_id]
        buildings_data = buildings_data[buildings_data.site_id.isin(site_id)]
        weather_data = weather_data[weather_data.site_id.isin(site_id)]
    if primary_use is not None:
        primary_use = primary_use if type(primary_use) in (list, tuple) \
            else [primary_use]
        buildings_data = buildings_data[
            buildings_data.primary_use.isin(primary_use)
        ]
    if meter is not None:
        meter = meter if type(meter) in (list, tuple) else [meter]
        meter_data = meter_data[meter_data.meter.isin(meter)]
    if meter_reading is not None:
        meter_data = meter_data[meter_data.meter_reading < meter_reading]

    buildings_ids = buildings_data.building_id.values
    meter_data = meter_data[meter_data.building_id.isin(buildings_ids)]

    return buildings_data, weather_data, meter_data


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
    merged_data = pd.merge(
        meter_building,
        weather_aux,
        on=['site_id', 'timestamp']
    )[1:]
    return merged_data


def preprocess_and_merge_data(
    buildings_data,
    weather_data,
    meter_data,
    buildings_fu,
    weather_fu,
    meter_fu,
    fit=False,
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
    :param fit: bool (default: False), whether union will only transform
        or also fit.
    :param merge_data_kwargs: dict, keyword arguments of `merge_data`
        function.
    :return:
        pandas.DataFrame, dataframe containing only target value
            (meter_reading).
        pandas.DataFrame, dataframe containing other columns.
    """
    # Preprocess data using corresponding feature unions
    buildings_features = buildings_fu.union_features(buildings_data, fit=fit)
    weather_features = weather_fu.union_features(weather_data, fit=fit)
    meter_features = meter_fu.union_features(meter_data, fit=fit)

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
