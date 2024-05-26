import logging
import datetime as dt
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from lightgbm import LGBMClassifier
import mlflow

import deployment.utils as utils
from model_registry.model_tags import ModelTags
from data.data_generator import DataGenerator
import settings
from deep_learning.neural_net import NeuralNet
from explainer.model_explainer import ModelExplainer

logging.basicConfig(level=logging.INFO)

class TrendType(Enum):
    UPTREND = 'uptrend'
    DOWNTREND = 'downtrend'


class DeploymentPipeline:
    def __init__(self, symbol: str, trend_type: TrendType) -> None:
        self.symbol = symbol
        self.trend_type = trend_type.value
        self._ml_flow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(f"{symbol}_{trend_type.value}_{dt.datetime.now().isoformat()}")
        self._evaluation_results = {
            'Classifier': [],
            'Accuracy': [],
            'Precision': [],
            'Positive_Accuracy': [],
            'Negative_Accuracy': [],
            'Overall_Score': [],
            'Run_Id': []
        }
        self._classifier_artifact_path = f'{symbol}_{trend_type.value}_classifier'
        self._registered_model_name = f"{symbol}_{trend_type.value}_model"
        self._prediction_window_days = settings.prediction_window_days
        self._target_pct = settings.target_uptrend_pct if trend_type == TrendType.UPTREND else settings.target_downtrend_pct
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_models(self) -> dict[str, object]:
        """
        This method trains the classifiers and stores the evaluation metrics in
        self._evaluation_results attribute.
        Returns:
        A dict with the name of the classifier and the trained model.
        Example:
        {
            "RandomForest": sklearn.ensemble.RandomForestClassifier,
            "XGBoost": xgb.XGBClassifier,
            "LightGBM": lightgbm.LGBMClassifier,
            "NeuralNet": deep_learning.neural_net.NeuralNet
        }
        """    
        class_weights = self._calculate_class_weights(self.y_train)
        scale_pos_weight=self.y_train.value_counts()[0] / self.y_train.value_counts()[1]

        classifiers = self._get_classifiers(class_weights=class_weights, scale_pos_weight=scale_pos_weight)
        for clf_name, clf in classifiers.items():
            with mlflow.start_run(run_name=f"{self.symbol}_{clf_name}") as run:
                metrics = utils.evaluate_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)
                if isinstance(clf, xgb.XGBClassifier):
                    mlflow.log_params({"scale_pos_weight": scale_pos_weight})
                else:
                    mlflow.log_params({"class_weights": class_weights})

                mlflow.log_metrics(metrics)
                signature = mlflow.models.infer_signature(self.X_test, clf.predict(self.X_test))
                
                if isinstance(clf, NeuralNet):
                    mlflow.tensorflow.log_model(
                        model=clf._model,
                        artifact_path=self._classifier_artifact_path
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=clf,
                        signature=signature,
                        artifact_path=self._classifier_artifact_path
                    )
            self._store_evaluation_results(classifier_name=clf_name, metrics=metrics, run_id=run.info.run_id)
        
        return classifiers

    def register_best_performing_model(self, classifiers: dict[str, object]) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """
        Stores the best performing model in the model registry
        Returns the version of the deployed model or None 
        in case that the best model failed to pass
        the performance thresholds.
        Params: 
        - classifiers: A dict with the name of the classifier and the trained model
        """
        results_df = pd.DataFrame(self._evaluation_results)
        results_df.sort_values(by=['Overall_Score'], ascending=False, inplace=True)
        results_df.reset_index(inplace=True)
        run_id = results_df['Run_Id'][0]
        classifier_name = results_df['Classifier'][0]

        # Check if best performing model is passing the performance thresholds
        positive_accuracy = results_df['Positive_Accuracy'][0]
        negative_accuracy = results_df['Negative_Accuracy'][0]
        overall_score = results_df['Overall_Score'][0]
        accuracy = results_df['Accuracy'][0]
        precision = results_df['Precision'][0]

        if positive_accuracy > 0.5 and negative_accuracy > 0.5 and overall_score > 0.6 and precision > 0.5:
            logging.info(f"Registering model for symbol: {self.symbol}")
            model_uri = f"runs:/{run_id}/{self._classifier_artifact_path}"            
            feature_importance_dict = self._get_feature_importance(
                classifier=classifiers[classifier_name]
            )
            tags = ModelTags(
                positive_accuracy=positive_accuracy,
                negative_accuracy=negative_accuracy,
                overall_score=overall_score,
                accuracy=accuracy,
                precision=precision,
                symbol=self.symbol,
                classifier=classifier_name,
                classified_trend=self.trend_type,
                target_pct=self._target_pct,
                prediction_window_days=self._prediction_window_days,
                feature_names=list(self.X_train.columns),
                feature_importance=feature_importance_dict
            )
            model_version = mlflow.register_model(model_uri=model_uri, name=self._registered_model_name, tags=tags.to_dict())
            return model_version
        else:
            logging.info(f"Model for {self.symbol} failed thresholds")
            return None

    def _get_feature_importance(
        self,
        classifier: object,
    ) -> dict[str, float]:
        """
        Returns the a dict with the mean shap value of each feature
        """
        explainer = ModelExplainer(model=classifier, sample_data=self.X_train)
        return explainer.explain(self.X_test)

    def run(self):
        self.create_train_test_sets()
        classifiers = self.train_models()
        self.register_best_performing_model(classifiers)

    def _store_evaluation_results(self, classifier_name: str, metrics: dict[str, float], run_id: str) -> None:
        self._evaluation_results['Classifier'].append(classifier_name)
        self._evaluation_results['Accuracy'].append(metrics['accuracy'])
        self._evaluation_results['Precision'].append(metrics['precision'])
        self._evaluation_results['Positive_Accuracy'].append(metrics['positive_accuracy'])
        self._evaluation_results['Negative_Accuracy'].append(metrics['negative_accuracy'])
        self._evaluation_results['Overall_Score'].append(metrics['overall_score'])
        self._evaluation_results['Run_Id'].append(run_id)


    def create_train_test_sets(
        self,
        training_data_pct: float = 0.95,
        target_col_name: str = 'target'
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple that contains:
        (X_train, y_train, X_test, y_test)
        """
        if self.trend_type == TrendType.DOWNTREND.value:
            dataset = DataGenerator(self.symbol).get_dataset(downtrend=True)
        else:
            dataset = DataGenerator(self.symbol).get_dataset(downtrend=False)

        train_dataset, test_dataset = utils.split_dataset(dataset, training_pct=training_data_pct)

        X_train = train_dataset.drop(columns=[target_col_name], axis=1)
        y_train = train_dataset[target_col_name]

        X_test = test_dataset.drop(columns=[target_col_name], axis=1)
        y_test = test_dataset[target_col_name]

        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train

        return (X_train, y_train, X_test, y_test)        


    def _calculate_class_weights(self, y_train: pd.DataFrame) -> dict[str, float]:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return dict(zip(classes, weights))


    def _get_classifiers(self, class_weights: dict[str, float], scale_pos_weight: float = None) -> dict[str, object]:
        return {
            "RandomForest": RandomForestClassifier(class_weight=class_weights),
            "SupportVectorMachine": SVC(probability=True, class_weight=class_weights),
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(class_weight=class_weights),
            "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME'),
            "RidgeClassifier": RidgeClassifier(class_weight=class_weights),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "MLPClassifier": MLPClassifier(),
            "LightGBM": LGBMClassifier(class_weight=class_weights),
            "NeuralNet": NeuralNet(class_weight=class_weights)
        }
