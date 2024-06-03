from dataclasses import dataclass
import time

import mlflow

from data.data_generator import DataGenerator
from deployment.deployment_pipeline import TrendType
from model_registry.model_tags import ModelTags
from model_registry.model_registry import ModelRegistry
import settings


@dataclass
class Prediction:
    symbol: str
    prediction: str
    tags: ModelTags


class BatchPredictions:
    def __init__(
        self,
        trend_type: TrendType,
        symbols: list[str] = None,
    ) -> None:
        self.trend_type = trend_type
        self.mlflow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
        self.predictions: list[Prediction] = []
        self.symbols = symbols
        mlflow.set_tracking_uri(settings.tracking_uri)

    def run(self) -> list[dict[str, str]]:
        count = 0
        # Get all registered models
        model_registry = ModelRegistry()
        for deployed_model in model_registry.get_deployed_models(self.trend_type, self.symbols):
            prediction_input = DataGenerator(deployed_model.symbol).get_prediction_input()
            count += 1

            self.predictions.append(
                Prediction(
                    symbol=deployed_model.symbol,
                    prediction=deployed_model.predict(prediction_input),
                    tags=deployed_model.tags,
                )
            )

            if count == 5:
                time.sleep(65)  # Provider limitation
                count = 0

        return self.predictions