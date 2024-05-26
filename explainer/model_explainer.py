import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from deep_learning.neural_net import NeuralNet


class ModelExplainer:
    def __init__(self, model: object, sample_data: pd.DataFrame) -> None:
        self.model = model
        self._explainer = self._create_explainer(model, sample_data)

    def _create_explainer(self, classifier: object, X: pd.DataFrame) -> shap.Explainer:    
        if isinstance(classifier, NeuralNet):
            return shap.KernelExplainer(
                model=classifier.predict_flatten,
                data=shap.utils.sample(X, 200),
                feature_names=list(X.columns)
            )
        
        if isinstance(classifier, XGBClassifier) or isinstance(classifier, LGBMClassifier):
            return shap.explainers.TreeExplainer(classifier)

        return shap.explainers.KernelExplainer(
            model=classifier.predict, 
            data=shap.utils.sample(X, 200),
            feature_names=list(X.columns)
        )

    def explain(self, X: pd.DataFrame) -> dict[str, float]:
        """
        Returns the a dict with the mean shap value of each feature
        """
        feature_importance_dict = dict()
        shap_values = self._explainer.shap_values(X)
        mean_shap_values = np.mean(shap_values, axis=0)
        for index, feature_name in enumerate(list(X.columns)):
            feature_importance_dict[feature_name] = mean_shap_values[index]

        return feature_importance_dict
