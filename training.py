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
import mlflow

from data_generator import DataGenerator
import utils

symbol = 'AVAX'

ml_flow_client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8080")
mlflow.set_tracking_uri("http://127.0.0.1:8080")

experiment_description = f"{symbol} Experiment"

experiment_tags = {
    "project_name": "crypto-forecasting",
    "mlflow.note.content": experiment_description,
}

# experiment = ml_flow_client.get_experiment(f"{symbol}_Models")

# if experiment is None:
#     experiment = ml_flow_client.create_experiment(
#         name=f"{symbol}_Models", tags=experiment_tags
#     )

mlflow.set_experiment(f"{symbol}_Models")

dataset = DataGenerator(symbol).get_dataset()

train_dataset, test_dataset = utils.split_dataset(dataset, training_pct=0.97)

X_train = train_dataset.drop(columns=['target'], axis=1)
y_train = train_dataset['target']

X_test = test_dataset.drop(columns=['target'], axis=1)
y_test = test_dataset['target']

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

classifiers = {
    "Random Forest": RandomForestClassifier(class_weight=class_weights),
    "Support Vector Machine": SVC(probability=True, class_weight=class_weights),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1]),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(class_weight=class_weights),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME'),
    "RidgeClassifier": RidgeClassifier(class_weight=class_weights),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier()
}

results = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Positive_Accuracy': [],
    'Negative_Accuracy': [],
    'Confusion Matrix': [],
    'Overall Score': [],
    "RunID": []
  }

for clf_name, clf in classifiers.items():
    with mlflow.start_run(run_name=f"{symbol}_{clf_name}") as run:
        # Evaluate classifier
        metrics = utils.evaluate_classifier(clf, X_train, y_train, X_test, y_test)
        mlflow.log_params({"class_weights": class_weights})
        mlflow.log_metrics({key: value for key, value in metrics.items() if key != 'cm'})
        mlflow.sklearn.log_model(
            sk_model=clf, input_example=X_train, artifact_path=f"{symbol}_{clf_name}"
        )

    # Store results
    results['Classifier'].append(clf_name)
    results['Accuracy'].append(metrics['accuracy'])
    results['Precision'].append(metrics['precision'])
    results['Positive_Accuracy'].append(metrics['positive_accuracy'])
    results['Negative_Accuracy'].append(metrics['negative_accuracy'])
    results['Confusion Matrix'].append(metrics['cm'])
    results['Overall Score'].append(metrics['overall_score'])
    results['RunID'].append(run.info.run_id)

results_df = pd.DataFrame(results)
results_df.sort_values(by=['Overall Score'], ascending=False, inplace=True)
results_df.reset_index(inplace=True)
print(results_df)


# Get best performing model
top_classifier_name = results_df['Classifier'][0]
top_run_id = results_df['RunID'][0]
classifier = classifiers[top_classifier_name]

# Check if best performing model is passing the performance thresholds
positive_accuracy = results_df['Positive_Accuracy'][0]
negative_accuracy = results_df['Negative_Accuracy'][0]

if positive_accuracy > 0.5 and negative_accuracy > 0.5:
    # Register model
    print(f"Registering model for symbol: {symbol}")
    top_clf_run = mlflow.get_run(top_run_id)
    model_uri = f"runs:/{top_clf_run.info.run_id}/{symbol}_{clf_name}"
    mv = mlflow.register_model(model_uri, f"{symbol}_{top_classifier_name}")
    print(f"Version: {mv.version}")
else:
    print(f"Model for {symbol} failed thresholds")

# Get prediction
# prediction_input = DataGenerator(symbol).get_prediction_input()
# if top_classifier_name == 'RidgeClassifier':
#     prediction = classifier.predict(prediction_input)
# else:
#     prediction = classifier.predict_proba(prediction_input)

# print(f"Prediction for {symbol}: {prediction}")