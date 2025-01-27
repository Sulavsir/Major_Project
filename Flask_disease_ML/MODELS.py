import numpy as np

# Implementing Logistic Regression from scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        # Gradient Descent
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / X.shape[0]) * np.dot(X.T, (predictions - y))
            db = (1 / X.shape[0]) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return (predictions >= 0.5).astype(int)

# Implementing Support Vector Machine (SVM) from scratch
class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        # Gradient Descent
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                if y[i] * (np.dot(X[i], self.weights) + self.bias) < 1:
                    # If the point is inside the margin, update the weights
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]
                else:
                    # Otherwise, just update based on the regularization term
                    self.weights -= self.learning_rate * 2 * self.lambda_param * self.weights

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.sign(linear_model)

# Implementing Random Forest from scratch
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        left_indices = X[:, best_split[0]] <= best_split[1]
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_split, left_tree, right_tree)

    def _find_best_split(self, X, y):
        # Implement splitting logic (e.g., based on Gini index )
        best_gini = float('inf')
        best_split = None
        for feature_index in range(X.shape[1]):
            values = set(X[:, feature_index])
            for value in values:
                left_indices = X[:, feature_index] <= value
                right_indices = ~left_indices
                left_y, right_y = y[left_indices], y[right_indices]

                gini = self._gini_index(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)
        return best_split

    def _gini_index(self, left_y, right_y):
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size
        gini_left = 1 - sum([(np.sum(left_y == c) / left_size) ** 2 for c in set(left_y)])
        gini_right = 1 - sum([(np.sum(right_y == c) / right_size) ** 2 for c in set(right_y)])
        return (left_size / total_size) * gini_left + (right_size / total_size) * gini_right

    def predict(self, X):
        return np.array([self._predict_instance(x, self.tree) for x in X])

    def _predict_instance(self, x, tree):
        if isinstance(tree, (float, int)):  # If we reached a leaf node
            return tree
        feature_index, value = tree[0]
        if x[feature_index] <= value:
            return self._predict_instance(x, tree[1])
        else:
            return self._predict_instance(x, tree[2])

#Random Forest Implementation
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))

