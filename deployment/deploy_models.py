import logging
import time
import warnings
import sys

import settings
from deployment.deployment_pipeline import DeploymentPipeline, TrendType

logging.basicConfig(level=logging.INFO)

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    count = 0
    trend_type = sys.argv[1]
    for crypto_symbol in settings.symbols:
        logging.info(f"Running deployment pipeline for {crypto_symbol}")
        try:
            pipeline = DeploymentPipeline(symbol=crypto_symbol, trend_type=TrendType(trend_type))
            pipeline.run()
            count += 1
        except Exception as e:
            logging.error(f"Error running deployment pipeline for {crypto_symbol}: {e}")

        if count == 5:
            time.sleep(65)  # Provider requests limitation is 30 requests per minute
            count = 0