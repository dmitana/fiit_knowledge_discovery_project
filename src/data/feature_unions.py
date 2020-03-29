from src.data.pipelines import site_id_pipeline, building_id_pipeline, \
    primary_use_pipeline, square_feet_pipeline
import pandas as pd
from sklearn.pipeline import FeatureUnion


class CustomFeatureUnion(FeatureUnion):
    def get_feature_names(self):
        """Get feature names from all pipelines."""
        feature_names = []
        for _, pipeline in self.transformer_list:
            _, transformer = list(pipeline.named_steps.items())[-1]
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
