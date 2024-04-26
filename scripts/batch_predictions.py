import time
import warnings

import mlflow

from data.data_generator import DataGenerator
import settings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    mlflow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    mlflow.set_tracking_uri(settings.tracking_uri)

    predictions = []

    count = 0
    # Get all registered models
    for model in mlflow_client.search_registered_models():
        model_name = model.name
        model_version = model.latest_versions[0].version
        tags = model.latest_versions[0].tags
        
        # Load the model
        try:
            model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
            symbol = tags['symbol']
            prediction_input = DataGenerator(symbol).get_prediction_input()
            count += 1

            # Get predictions
            predictions.append(
                {
                    "symbol": symbol,
                    "prediction": model.predict(prediction_input),
                    "tags": tags
                }
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

        if count == 5:
            time.sleep(65)  # Provider requests limitation is 30 requests per minute
            count = 0            

    for prediction in predictions:
        print(prediction)
        print('--------------------------')