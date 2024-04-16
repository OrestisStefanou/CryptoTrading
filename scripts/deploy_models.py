import logging
import time

import settings
from deployment_pipeline import DeploymentPipeline

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    count = 0
    for crypto_symbol in settings.symbols:
        logging.info(f"Running deployment pipeline for {crypto_symbol}")
        try:
            pipeline = DeploymentPipeline(symbol=crypto_symbol)
            pipeline.run()
            count += 1
        except Exception as e:
            logging.error(f"Error running deployment pipeline for {crypto_symbol}: {e}")

        if count == 2:
            time.sleep(65)  # Provider requests limitation is 30 requests per minute
            count = 0