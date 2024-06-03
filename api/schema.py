from pydantic import BaseModel

from deployment.deployment_pipeline import TrendType


class PerformanceMetrics(BaseModel):
    positive_accuracy: float
    negative_accuracy: float
    overall_score: float
    accuracy: float
    precision: float    


class DeployedModel(BaseModel):
    symbol: str
    trend_type: TrendType
    feature_importance: dict[str, float]
    performance_metrics: PerformanceMetrics
    target_pct: float
    prediction_window_days: int


class Prediction(BaseModel):
    prediction_probabilities: float
    symbol: str
    trend_type: TrendType
