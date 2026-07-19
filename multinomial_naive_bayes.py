import numpy as np
import pandas as pd


class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._classes = None
        self._priors = None
        self._likelihoods = None

    def fit(self, X_train_input, y_train_input):
        # Convert features to a dense NumPy array
        if hasattr(X_train_input, "toarray"):
            X_train = X_train_input.toarray()
        elif isinstance(X_train_input, pd.DataFrame):
            X_train = X_train_input.to_numpy()
        else:
            X_train = np.asarray(X_train_input)

        # Convert labels to a NumPy array
        if isinstance(y_train_input, pd.Series):
            y_train = y_train_input.to_numpy()
        else:
            y_train = np.asarray(y_train_input)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a two-dimensional array")

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have the same number of samples"
            )

        number_of_samples, number_of_features = X_train.shape

        self._classes = np.unique(y_train)
        number_of_classes = len(self._classes)

        self._priors = np.zeros(number_of_classes)
        self._likelihoods = np.zeros(
            (number_of_classes, number_of_features)
        )

        for index, class_label in enumerate(self._classes):
            X_train_class = X_train[y_train == class_label]

            # Prior probability P(class)
            self._priors[index] = (
                X_train_class.shape[0] / number_of_samples
            )

            # Total count of each word for this class
            feature_counts = X_train_class.sum(axis=0)

            # Laplace-smoothed likelihood P(word | class)
            self._likelihoods[index] = (
                feature_counts + self.alpha
            ) / (
                feature_counts.sum()
                + self.alpha * number_of_features
            )

        return self

    def predict(self, X_test_input):
        if self._classes is None:
            raise RuntimeError(
                "The classifier has not been trained. Call fit() first."
            )

        if hasattr(X_test_input, "toarray"):
            X_test = X_test_input.toarray()
        elif isinstance(X_test_input, pd.DataFrame):
            X_test = X_test_input.to_numpy()
        else:
            X_test = np.asarray(X_test_input)

        # Allow one sample represented as a one-dimensional array
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        return np.array([
            self._predict(row)
            for row in X_test
        ])

    def _predict(self, x_test):
        posteriors = []

        for index, class_label in enumerate(self._classes):
            log_prior = np.log(self._priors[index])

            log_likelihood = np.sum(
                np.log(self._likelihoods[index]) * x_test
            )

            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        best_class_index = np.argmax(posteriors)

        return self._classes[best_class_index]

    def score(self, X_test, y_test):
        y_test = np.asarray(y_test)
        y_predicted = self.predict(X_test)

        return np.mean(y_predicted == y_test)