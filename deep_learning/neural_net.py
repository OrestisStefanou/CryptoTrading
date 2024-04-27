import pandas as pd
import tensorflow as tf

class NeuralNet:
    """
    A wrapper class on top of tensorflow
    """
    def __init__(self, class_weight: dict[str, float]) -> None:
        self._class_weight = class_weight
        self.classes_ = [0, 1]
        self._model = None
        self._normalizer = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self._normalizer = tf.keras.layers.Normalization()
        self._normalizer.adapt(X_train.to_numpy())

        # Build the neural network
        self._model = tf.keras.Sequential([
            self._normalizer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self._model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Fit the model
        self._model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=100, class_weight=self._class_weight)

    def predict_proba(self, X_test: pd.DataFrame) -> list[list[float, float]]:
        """
        Returns the prediction probabilities in a list.
        At index 0 are the prediction probabilities for the zero class
        At index 1 are the prediction probabilities for the one class
        """
        y_pred_probs = self._model.predict(X_test.to_numpy())
        return [
            [1 - prob, prob] for prob in y_pred_probs
        ]

    def predict(self, X_test: pd.DataFrame) -> list[float]:
        return self._model.predict(X_test.to_numpy())