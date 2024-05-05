import mlflow
from mlflow.entities.model_registry import RegisteredModel

class DeployedModel:
    def __init__(self, model: RegisteredModel) -> None:
        self.model_name = model.name
        self.model_version = model.latest_versions[0].version
        self.tags = model.latest_versions[0].tags
        self.classified_trend = self.tags['classified_trend']
        self.symbol = self.tags['symbol']
        self.classifier_name = self.tags['classifier']

        model_uri = f'models:/{self.model_name}/{self.model_version}'
        if self.classifier_name == 'NeuralNet':
            self.model = mlflow.tensorflow.load_model(model_uri=model_uri)
        else:
            self.model = mlflow.sklearn.load_model(model_uri=model_uri)
    
    def predict(self, model_input) -> float:
        """
        Returns the prediction probabilities for the positive class

        Params:
        - model_input: The input that the deployed model expects 
        """
        if self.classifier_name == 'NeuralNet':
            return self.model.predict(model_input)[0][0]
        else:
            if self.classifier_name == 'RidgeClassifier':
                return self.model.predict(model_input)[0]
            else:
                return self.model.predict_proba(model_input)[0][1]        
