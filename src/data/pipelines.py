from src.data.preprocessing.transformers import PrimaryUseTransformer, \
    FeatureSelectorTransformer, RollingAverageNanTransformer, \
    OutlierTransformer, ValuePicker, OneHotEncoderTransformer, \
    StandardScalerTransformer, AddPreviousMeterReadingTransformer
from sklearn.pipeline import Pipeline

site_id_pipeline = Pipeline([
    ('select feature', FeatureSelectorTransformer('site_id'))
])

building_id_pipeline = Pipeline([
    ('select feature', FeatureSelectorTransformer('building_id'))
])

timestamp_pipeline = Pipeline([
    ('select feature', FeatureSelectorTransformer('timestamp'))
])

primary_use_pipeline = Pipeline([
    ('merge categories', PrimaryUseTransformer()),
    ('one hot encoding', OneHotEncoderTransformer('primary_use'))
])

square_feet_pipeline = Pipeline([
    (
        'normalization',
        StandardScalerTransformer('square_feet', group_by_column='building_id')
    )
])

air_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('air_temperature')),
    (
        'normalization',
        StandardScalerTransformer(
            'air_temperature',
            group_by_column=['site_id', 'timestamp']
        )
    )
])

air_temperature_without_outliers_pipeline = Pipeline([
    (
        'replace outliers',
        OutlierTransformer(
            'air_temperature',
            group_by_columns=['site_id', 'timestamp']
        )
    ),
    ('air temperature', air_temperature_pipeline)
])

dew_temperature_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('dew_temperature')),
    (
        'normalization',
        StandardScalerTransformer(
            'dew_temperature',
            group_by_column=['site_id', 'timestamp']
        )
    )
])

dew_temperature_without_outliers_pipeline = Pipeline([
    (
        'replace outliers',
        OutlierTransformer(
            'dew_temperature',
            group_by_columns=['site_id', 'timestamp']
        )
    ),
    ('dew temperature', dew_temperature_pipeline)
])

sea_level_pressure_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('sea_level_pressure')),
    (
        'normalization',
        StandardScalerTransformer(
            'sea_level_pressure',
            group_by_column=['site_id', 'timestamp']
        )
    )
])

wind_speed_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_speed')),
    (
        'normalization',
        StandardScalerTransformer(
            'wind_speed',
            group_by_column=['site_id', 'timestamp']
        )
    )
])

wind_speed_without_outliers_pipeline = Pipeline([
    (
        'replace outliers',
        OutlierTransformer(
            'wind_speed',
            group_by_columns=['site_id', 'timestamp']
        )
    ),
    ('wind speed', wind_speed_pipeline)
])

wind_direction_pipeline = Pipeline([
    ('rolling average', RollingAverageNanTransformer('wind_direction')),
    (
        'normalization',
        StandardScalerTransformer(
            'wind_direction',
            group_by_column=['site_id', 'timestamp']
        )
    )
])

meter_pipeline = Pipeline([
    (
        'normalization',
        StandardScalerTransformer(
            'meter_reading',
            new_column='meter_reading_scaled',
            all_columns=True
        )
    ),
    ('previous meter readings',
     AddPreviousMeterReadingTransformer(time_horizon=5, sample_length=3600))
])

building_id_meter_pipeline = Pipeline([
    ('meter type', ValuePicker(feature='meter', specific_value=0)),
    ('meter reading', ValuePicker(feature='meter_reading', threshold=200)),
    ('select feature', FeatureSelectorTransformer('building_id'))
])

timestamp_meter_pipeline = Pipeline([
    ('meter type', ValuePicker(feature='meter', specific_value=0)),
    ('meter reading', ValuePicker(feature='meter_reading', threshold=200)),
    ('select feature', FeatureSelectorTransformer('timestamp'))
])
