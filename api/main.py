import ast

from fastapi import FastAPI, HTTPException

from model_registry.model_registry import ModelRegistry
from deployment.deployment_pipeline import TrendType
from api import schema
from predictions.batch_predict import BatchPredictions

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


@app.get("/prediction", status_code=200)
async def get_prediction(symbol: str, trend_type: TrendType) -> schema.Prediction:
    batch_predictions = BatchPredictions(
        trend_type=trend_type,
        symbols=[symbol, ]
    ).run(store_in_db=False)

    if len(batch_predictions) == 0:
        raise HTTPException(status_code=404, detail=f"Model for symbol {symbol} and trend type {trend_type} not found.")
    
    return schema.Prediction(
        prediction_probabilities=batch_predictions[0].prediction,
        symbol=batch_predictions[0].symbol,
        trend_type=batch_predictions[0].tags.classified_trend
    )
