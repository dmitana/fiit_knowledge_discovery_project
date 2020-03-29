from src.data.preprocessing import PrimaryUseTransformer, \
    FeatureSelectorTransformer, RollingAverageNanTransformer, \
    OutlierTransformer, ValuePicker, OneHotEncoderTransformer, \
    StandardScalerTransformer
from sklearn.pipeline import Pipeline

site_id_pipeline = Pipeline([
    ('select feature', FeatureSelectorTransformer('site_id'))
])

building_id_pipeline = Pipeline([
    ('select feature', FeatureSelectorTransformer('building_id'))
])

primary_use_pipeline = Pipeline([
    ('merge categories', PrimaryUseTransformer()),
    ('one hot encoding', OneHotEncoderTransformer('primary_use'))
])

square_feet_pipeline = Pipeline([
    ('normalization', StandardScalerTransformer('square_feet'))
])

air_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('air_temperature')),
    ('normalization', StandardScalerTransformer('air_temperature'))
])

air_temperature_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('air_temperature')),
    ('air temperature', air_temperature_pipeline)
])

dew_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('dew_temperature')),
    ('normalization', StandardScalerTransformer('dew_temperature'))
])

dew_temperature_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('dew_temperature')),
    ('dew temperature', dew_temperature_pipeline)
])

sea_level_pressure_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('sea_level_pressure')),
    ('normalization', StandardScalerTransformer('sea_level_pressure'))
])

wind_speed_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_speed')),
    ('normalization', StandardScalerTransformer('wind_speed'))
])

wind_speed_without_outliers_pipeline = Pipeline([
    ('replace outliers', OutlierTransformer('wind_speed')),
    ('wind speed', wind_speed_pipeline)
])

wind_direction_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_direction')),
    ('normalization', StandardScalerTransformer('wind_direction'))
])

meter_pipeline = Pipeline([
    ('meter type', ValuePicker(feature='meter', specific_value=0)),
    ('meter reading', ValuePicker(feature='meter_reading', threshold=200)),
    ('feature select', FeatureSelectorTransformer('meter_reading')),
    ('normalization', StandardScaler())
])
