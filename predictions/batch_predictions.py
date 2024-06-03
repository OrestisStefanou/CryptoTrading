import warnings
import sys

from deployment.deployment_pipeline import TrendType
from predictions.batch_predict import BatchPredictions

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    trend_type = TrendType(sys.argv[1])
    
    predictions = BatchPredictions(trend_type).run()

    for prediction in predictions:
        print(prediction)
        print('--------------------------')