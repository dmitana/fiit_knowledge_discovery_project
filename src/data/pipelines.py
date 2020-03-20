from src.data.preprocessing import PrimaryUseTransformer, \
    FeatureSelectorTransformer
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
