import mlflow

import settings
from model_registry.deployed_model import DeployedModel
from deployment.deployment_pipeline import TrendType


class ModelRegistry:
    def __init__(self) -> None:
        self.mlflow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
        mlflow.set_tracking_uri(settings.tracking_uri)

    def get_deployed_models(
        self,
        trend_type: TrendType = None,
        symbols: list[str] = None
    ) -> list[DeployedModel]:
        deployed_models = []

        for model in self.mlflow_client.search_registered_models():
            deployed_model = DeployedModel(model)

            if symbols and deployed_model.symbol not in symbols:
                continue

            if trend_type and deployed_model.classified_trend != trend_type.value:
                continue
            
            deployed_models.append(deployed_model)
        
        return deployed_models
