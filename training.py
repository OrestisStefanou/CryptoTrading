import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from data_generator import DataGenerator
import utils

symbol = 'AVAX'
dataset = DataGenerator(symbol).get_dataset()

n = len(dataset)
train_dataset = dataset[0:int(n*0.97)]
test_dataset = dataset[int(n*0.97):]

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

# Initialize lists to store results
results = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Positive_Accuracy': [],
    'Negative_Accuracy': [],
    'Confusion Matrix': [],
    'Overall Score': []
  }

# Iterate through classifiers
for clf_name, clf in classifiers.items():
    # Evaluate classifier
    metrics = utils.evaluate_classifier(clf, X_train, y_train, X_test, y_test)

    # Store results
    results['Classifier'].append(clf_name)
    results['Accuracy'].append(metrics['accuracy'])
    results['Precision'].append(metrics['precision'])
    results['Positive_Accuracy'].append(metrics['positive_accuracy'])
    results['Negative_Accuracy'].append(metrics['negative_accuracy'])
    results['Confusion Matrix'].append(metrics['cm'])
    results['Overall Score'].append(metrics['overall_score'])


results_df = pd.DataFrame(results)
results_df.sort_values(by=['Overall Score'], ascending=False, inplace=True)
results_df.reset_index(inplace=True)
print(results_df)


# Get best performing model
top_classifier_name = results_df['Classifier'][0]
classifier = classifiers[top_classifier_name]

# Get prediction
prediction_input = DataGenerator(symbol).get_prediction_input()
if top_classifier_name in ['RidgeClassifier', 'Neural Net']:
    if top_classifier_name == 'Neural Net':
        pass
        # prediction = classifier.predict(scaler.transform(prediction_input))
    else:
        prediction = classifier.predict(prediction_input)
else:
    prediction = classifier.predict_proba(prediction_input)

print(f"Prediction for {symbol}: {prediction}")