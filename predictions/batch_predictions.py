import time
import warnings
import sys

import mlflow

from data.data_generator import DataGenerator
from deployment.deployment_pipeline import TrendType
from model_registry.deployed_model import DeployedModel
import settings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    trend_type = TrendType(sys.argv[1])
    mlflow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    mlflow.set_tracking_uri(settings.tracking_uri)

    predictions = []

    count = 0
    # Get all registered models
    for model in mlflow_client.search_registered_models():
        print(model)
        print('---------------------------------')
        continue
        deployed_model = DeployedModel(model)        
        # Load the model
        try:
            if deployed_model.classified_trend != trend_type.value:
                continue

            prediction_input = DataGenerator(deployed_model.symbol).get_prediction_input()
            count += 1

            # Get predictions
            predictions.append(
                {
                    "symbol": deployed_model.symbol,
                    "prediction": deployed_model.predict(prediction_input),
                    "tags": deployed_model.tags
                }
            )
        except Exception as e:
            print(f"Error loading model {model.name}: {e}")

        if count == 5:
            time.sleep(65)  # Provider requests limitation is 30 requests per minute
            count = 0            

    for prediction in predictions:
        print(prediction)
        print('--------------------------')