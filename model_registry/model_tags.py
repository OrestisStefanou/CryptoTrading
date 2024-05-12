from dataclasses import dataclass, asdict

@dataclass
class ModelTags:
    positive_accuracy: float
    negative_accuracy: float
    overall_score: float
    accuracy: float
    precision: float
    symbol: str
    classifier: str
    classified_trend: str
    target_pct: float
    prediction_window_days: int
    feature_names: list[str]

    def to_dict(self):
        return asdict(self)
