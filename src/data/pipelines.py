from src.data.preprocessing import PrimaryUseTransformer, \
    FeatureSelectorTransformer, RollingAverageNanTransformer, \
    OutlierTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

primary_use_pipeline = Pipeline([
    ('merge categories', PrimaryUseTransformer()),
    ('feature select', FeatureSelectorTransformer('primary_use')),
    ('one hot encoding', OneHotEncoder(sparse=False))
])

square_feet_pipeline = Pipeline([
    ('feature select', FeatureSelectorTransformer('square_feet')),
    ('normalization', StandardScaler())
])

air_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('air_temperature')),
    ('feature select', FeatureSelectorTransformer('air_temperature')),
    ('normalization', StandardScaler())
])

air_temperature_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('air_temperature')),
    ('air temperature', air_temperature_pipeline)
])

dew_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('dew_temperature')),
    ('feature select', FeatureSelectorTransformer('dew_temperature')),
    ('normalization', StandardScaler())
])

dew_temperature_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('dew_temperature')),
    ('dew temperature', dew_temperature_pipeline)
])

sea_level_pressure_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('sea_level_pressure')),
    ('feature select', FeatureSelectorTransformer('sea_level_pressure')),
    ('normalization', StandardScaler())
])

wind_speed_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_speed')),
    ('feature select', FeatureSelectorTransformer('wind_speed')),
    ('normalization', StandardScaler())
])

wind_speed_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('wind_speed')),
    ('wind speed', wind_speed_pipeline)
])

wind_direction_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_direction')),
    ('feature select', FeatureSelectorTransformer('wind_direction')),
    ('normalization', StandardScaler())
])
