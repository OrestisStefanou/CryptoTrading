import ast

from fastapi import FastAPI

from model_registry.model_registry import ModelRegistry
from deployment.deployment_pipeline import TrendType
from api import schema

app = FastAPI()

@app.get("/models", status_code=200)
async def get_models() -> list[schema.DeployedModel]:
    deployed_models = ModelRegistry().get_deployed_models()
    return [
        schema.DeployedModel(
            symbol=model.symbol,
            trend_type=TrendType(model.classified_trend),
            # Convert string to dict
            feature_importance=ast.literal_eval(model.tags.feature_importance),
            target_pct=model.tags.target_pct,
            prediction_window_days=model.tags.prediction_window_days,
            performance_metrics=schema.PerformanceMetrics(
                positive_accuracy=model.tags.positive_accuracy,
                negative_accuracy=model.tags.negative_accuracy,
                overall_score=model.tags.overall_score,
                accuracy=model.tags.accuracy,
                precision=model.tags.precision
            )
        )
        for model in deployed_models
    ]
