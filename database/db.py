import duckdb

import settings

class Database:
    def __init__(self, config: dict = None, read_only: bool = False) -> None:
        self._config = config if config else {}
        self._read_only = read_only
    
    def store_predictions(
        self,
        symbol: str,
        model_name: str,
        model_version: int,
        prediction_prob: float,
        prediction_input: dict,
        target_pct: float,
        prediction_window_days: int
    ) -> None:
        with duckdb.connect(database=settings.database_name, read_only=self._read_only, config=self._config) as con:
            con.execute(
                """
                INSERT INTO model_predictions VALUES (current_date(), ?, ?, ?, ?, ?, ?, ?)
                """,
                [symbol, model_name, model_version, prediction_prob, prediction_input, target_pct, prediction_window_days]
            )
