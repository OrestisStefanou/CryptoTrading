import logging
import datetime as dt

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
from data.data_generator import DataGenerator
import settings
from deep_learning.neural_net import NeuralNet

logging.basicConfig(level=logging.INFO)

class DeploymentPipeline:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._ml_flow_client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(f"{symbol}_Models_{dt.datetime.now().isoformat()}")   # Update this
        self._evaluation_results = {
            'Classifier': [],
            'Accuracy': [],
            'Precision': [],
            'Positive_Accuracy': [],
            'Negative_Accuracy': [],
            'Overall_Score': [],
            'Run_Id': []
        }
        self._artifact_path: str = f'{symbol}_classifier'
        self._registered_model_name: str = f"{symbol}_model"

    def train_models(self, training_data_pct: float = 0.97) -> None:
        X_train, y_train, X_test, y_test = self._create_train_test_sets(
            training_data_pct=training_data_pct
        )
        class_weights = self._calculate_class_weights(y_train)
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1]
        logging.info(f"Class weights: {class_weights}")

        classifiers = self._get_classifiers(class_weights=class_weights, scale_pos_weight=scale_pos_weight)
        for clf_name, clf in classifiers.items():
            with mlflow.start_run(run_name=f"{self.symbol}_{clf_name}") as run:
                metrics = utils.evaluate_classifier(clf, X_train, y_train, X_test, y_test)
                if isinstance(clf, xgb.XGBClassifier):
                    mlflow.log_params({"scale_pos_weight": scale_pos_weight})
                else:
                    mlflow.log_params({"class_weights": class_weights})

                mlflow.log_metrics(metrics)
                signature = mlflow.models.infer_signature(X_test, clf.predict(X_test))
                
                if isinstance(clf, NeuralNet):
                    mlflow.tensorflow.log_model(
                        model=clf._model,
                        artifact_path=self._artifact_path
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=clf,
                        signature=signature,
                        artifact_path=self._artifact_path
                    )
            self._store_evaluation_results(classifier_name=clf_name, metrics=metrics, run_id=run.info.run_id)


    def register_best_performing_model(self) -> None:
        results_df = pd.DataFrame(self._evaluation_results)
        results_df.sort_values(by=['Overall_Score'], ascending=False, inplace=True)
        results_df.reset_index(inplace=True)
        run_id = results_df['Run_Id'][0]

        # Check if best performing model is passing the performance thresholds
        positive_accuracy = results_df['Positive_Accuracy'][0]
        negative_accuracy = results_df['Negative_Accuracy'][0]
        overall_score = results_df['Overall_Score'][0]
        accuracy = results_df['Accuracy'][0]
        precision = results_df['Precision'][0]

        if positive_accuracy > 0.5 and negative_accuracy > 0.5 and overall_score > 0.5:
            logging.info(f"Registering model for symbol: {self.symbol}")
            model_uri = f"runs:/{run_id}/{self._artifact_path}"
            # Store the performance of the metrics in the tags
            tags = {
                "positive_accuracy": positive_accuracy,
                "negative_accuracy": negative_accuracy,
                "overall_score": overall_score,
                "accuracy": accuracy,
                "precision": precision,
                "symbol": self.symbol,
            }
            mlflow.register_model(model_uri=model_uri, name=self._registered_model_name, tags=tags)
        else:
            logging.info(f"Model for {self.symbol} failed thresholds")


    def run(self):
        self.train_models()
        self.register_best_performing_model()


    def _store_evaluation_results(self, classifier_name: str, metrics: dict[str, float], run_id: str) -> None:
        self._evaluation_results['Classifier'].append(classifier_name)
        self._evaluation_results['Accuracy'].append(metrics['accuracy'])
        self._evaluation_results['Precision'].append(metrics['precision'])
        self._evaluation_results['Positive_Accuracy'].append(metrics['positive_accuracy'])
        self._evaluation_results['Negative_Accuracy'].append(metrics['negative_accuracy'])
        self._evaluation_results['Overall_Score'].append(metrics['overall_score'])
        self._evaluation_results['Run_Id'].append(run_id)


    def _create_train_test_sets(
        self,
        training_data_pct: float = 0.95,
        target_col_name: str = 'target'
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple that contains:
        (X_train, y_train, X_test, y_test)
        """
        dataset = DataGenerator(self.symbol).get_dataset()
        train_dataset, test_dataset = utils.split_dataset(dataset, training_pct=training_data_pct)

        X_train = train_dataset.drop(columns=[target_col_name], axis=1)
        y_train = train_dataset[target_col_name]

        X_test = test_dataset.drop(columns=[target_col_name], axis=1)
        y_test = test_dataset[target_col_name]

        return(X_train, y_train, X_test, y_test)        


    def _calculate_class_weights(self, y_train: pd.DataFrame) -> dict[str, float]:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return dict(zip(classes, weights))


    def _get_classifiers(self, class_weights: dict[str, float], scale_pos_weight: float = None) -> dict[str, object]:
        return {
            "Random Forest": RandomForestClassifier(class_weight=class_weights),
            "Support Vector Machine": SVC(probability=True, class_weight=class_weights),
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(class_weight=class_weights),
            "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME'),
            "RidgeClassifier": RidgeClassifier(class_weight=class_weights),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "MLPClassifier": MLPClassifier(),
            "LightGBM": LGBMClassifier(class_weight=class_weights),
            "NeuralNet": NeuralNet(class_weight=class_weights)
        }
