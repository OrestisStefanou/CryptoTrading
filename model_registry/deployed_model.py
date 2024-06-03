import mlflow
from mlflow.entities.model_registry import RegisteredModel
import pandas as pd

from database.db import Database
from deployment.deployment_pipeline import TrendType
from deep_learning.neural_net import NeuralNet
from model_registry.model_tags import ModelTags
import settings

class DeployedModel:
    def __init__(self, model: RegisteredModel) -> None:
        self.model_name = model.name
        self.model_version = model.latest_versions[0].version
        self.tags: ModelTags = ModelTags(**model.latest_versions[0].tags)
        self.classified_trend = self.tags.classified_trend
        self.symbol = self.tags.symbol
        self.classifier_name = self.tags.classifier

        model_uri = f'models:/{self.model_name}/{self.model_version}'
        if self.classifier_name == 'NeuralNet':
            self.model = NeuralNet(
                model=mlflow.tensorflow.load_model(model_uri=model_uri)
            )
        else:
            self.model = mlflow.sklearn.load_model(model_uri=model_uri)
    
    def predict(self, model_input: pd.DataFrame, store_in_db: bool = True) -> float:
        """
        Returns the prediction probabilities for the positive class

        Params:
        - model_input: The input that the deployed model expects 
        """
        if self.classifier_name == 'RidgeClassifier':
            prediction =  self.model.predict(model_input)[0]
        else:
            prediction =  self.model.predict_proba(model_input)[0][1]        

        if store_in_db:
            self._store_predictions(prediction_prob=float(prediction), model_input=model_input.to_dict('records'))
        
        return prediction

    def _store_predictions(self, prediction_prob: float, model_input: dict) -> None:
        target_pct = self.tags.target_pct
        if target_pct is None:
            if self.classified_trend == TrendType.UPTREND.value:
                target_pct = settings.target_uptrend_pct
            else:
                target_pct = settings.target_downtrend_pct

        prediction_window_days = self.tags.prediction_window_days
        if prediction_window_days is None:
            prediction_window_days = settings.prediction_window_days

        Database().store_predictions(
            symbol=self.symbol,
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_prob=prediction_prob,
            prediction_input=model_input,
            target_pct=target_pct,
            prediction_window_days=prediction_window_days
        )
