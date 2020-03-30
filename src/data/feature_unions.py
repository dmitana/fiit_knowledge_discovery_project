from src.data.pipelines import site_id_pipeline, building_id_pipeline, \
    primary_use_pipeline, square_feet_pipeline, air_temperature_pipeline, \
    air_temperature_without_outliers_pipeline, dew_temperature_pipeline, \
    dew_temperature_without_outliers_pipeline, wind_direction_pipeline, \
    wind_speed_pipeline, wind_speed_without_outliers_pipeline, \
    timestamp_pipeline, meter_pipeline, building_id_meter_pipeline, \
    timestamp_meter_pipeline
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline


class CustomFeatureUnion(FeatureUnion):
    def get_feature_names(self):
        """Get feature names from all pipelines."""
        feature_names = []
        for _, pipeline in self.transformer_list:
            # There can be nested Pipelines so we have to iterate to
            # transformer
            _, transformer = list(pipeline.named_steps.items())[-1]
            while isinstance(transformer, Pipeline):
                _, transformer = list(transformer.named_steps.items())[-1]
            names = transformer.get_feature_names()
            feature_names.extend(names)
        return feature_names

    def union_features(self, x):
        """
        Union features and assign them names.

        :param x: pandas.DataFrame, dataframe to get features from.
        :return: pandas.DataFrame, dataframe containing transformed
            feature.
        """
        features = pd.DataFrame(self.fit_transform(x))
        features.columns = self.get_feature_names()
        return features


buildings_fu = CustomFeatureUnion(
    [
        ('site id', site_id_pipeline),
        ('building_id', building_id_pipeline),
        ('primary use', primary_use_pipeline),
        ('square feet', square_feet_pipeline)
    ],
    n_jobs=-1
)

weather_fu = CustomFeatureUnion(
    [
        ('site id', site_id_pipeline),
        ('timestamp', timestamp_pipeline),
        ('air temperature', air_temperature_pipeline),
        ('dew temperature', dew_temperature_pipeline),
        ('wind direction', wind_direction_pipeline),
        ('wind speed', wind_speed_pipeline)
    ],
    n_jobs=-1
)

weather_without_outliers_fu = CustomFeatureUnion(
    [
        ('site id', site_id_pipeline),
        ('timestamp', timestamp_pipeline),
        ('air temperature', air_temperature_without_outliers_pipeline),
        ('dew temperature', dew_temperature_without_outliers_pipeline),
        ('wind direction', wind_direction_pipeline),
        ('wind speed', wind_speed_without_outliers_pipeline)
    ],
    n_jobs=-1
)

meter_fu = CustomFeatureUnion(
    [
        ('building id', building_id_meter_pipeline),
        ('timestamp', timestamp_meter_pipeline),
        ('meter', meter_pipeline)
    ],
    n_jobs=-1
)
