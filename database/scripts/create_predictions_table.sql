CREATE TABLE model_predictions (
    created_at DATE,
    symbol VARCHAR,
    model_name VARCHAR,
    model_version INTEGER,
    prediction_prob DECIMAL,
    prediction_input JSON,
    target_pct DECIMAL,
    prediction_window_days INTEGER
);