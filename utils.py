import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.linear_model import RidgeClassifier

def split_dataset(dataset: pd.DataFrame, training_pct: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(dataset)
    train_dataset = dataset[0:int(n*training_pct)]
    test_dataset = dataset[int(n*training_pct):]

    return train_dataset, test_dataset


def get_overall_score(accuracy: float, precision: float, negative_accuracy: float, positive_accuracy: float) -> float:
    return (0.3 * accuracy) + (0.3 * precision) + (0.1 * negative_accuracy) + (0.3 * positive_accuracy)


def evaluate_classifier(
    classifier: object,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    threshold: float = 0.5
):
    """
    Returns a dictionary with the following metrics:
    - accuracy
    - precision
    - positive_accuracy
    - negative_accuracy
    - cm (confusion matrix)
    - overall_score
    """
    classifier.fit(X_train, y_train)
    labels =  classifier.classes_
    if labels[0] != 0:
        raise Exception("LABELS ARE FUCKED UP")

    if not isinstance(classifier, RidgeClassifier):
      y_prob = classifier.predict_proba(X_test)
      y_pred = [
          1 if prediction_prob[1] > threshold else 0
          for prediction_prob in y_prob
      ]
    else:
      y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    negative_accuracy = cm[0][0] / (cm[0][0] + cm[0][1])
    positive_accuracy = cm[1][1] / (cm[1][0] + cm[1][1])
    return {
        "accuracy": accuracy,
        "precision": precision,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "cm": cm,
        "true_negatives": cm[0][0],
        "true_positives": cm[1][1],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "overall_score": get_overall_score(accuracy, precision, negative_accuracy, positive_accuracy)
    }
