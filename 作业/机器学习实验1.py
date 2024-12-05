import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, class_labels, feature_index=None, threshold=None,
                 left=None, right=None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.class_labels = class_labels
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf_node(self):
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, self.max_depth)

    def build_tree(self, X, y, depth):
        if len(X) <= 1 or depth >= self.max_depth:
            return self.build_leaf(X, y)

        best_gini = 1
        best_idx, best_thr = None, None
        num_parent = [float(sum(y == c)) for c in np.unique(y)]
        parent_gini = 1.0 - sum((num / len(y)) ** 2 for num in num_parent)

        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                num_left, num_right = self.split(X, y, idx, thr)
                gini_left = self.gini(y[:num_left])
                gini_right = self.gini(y[num_left:])
                gini = (num_left / len(y)) * gini_left + (num_right / len(y)) * gini_right
                if gini < best_gini:
                    best_gini = gini
                    best_idx, best_thr = idx, thr

        if best_idx is not None:
            left = self.build_tree(X[X[:, best_idx] < best_thr], y[X[:, best_idx] < best_thr], depth + 1)
            right = self.build_tree(X[X[:, best_idx] >= best_thr], y[X[:, best_idx] >= best_thr], depth + 1)
            return Node(parent_gini, len(y), num_parent, np.unique(y).tolist(), feature_index=best_idx,
                        threshold=best_thr, left=left, right=right)
        else:
            return self.build_leaf(X, y)

    def split(self, X, y, idx, thr):
        left = X[X[:, idx] < thr]
        right = X[X[:, idx] >= thr]
        y_left = y[X[:, idx] < thr]
        y_right = y[X[:, idx] >= thr]
        return len(y_left), len(y_right)

    def gini(self, y):
        if len(y) == 0:
            return 0

        y_val_counts = np.bincount(y)
        gini = 1.0 - sum((v / len(y)) ** 2 for v in y_val_counts)
        return gini

    def build_leaf(self, X, y):
        class_labels = np.unique(y)
        num_samples_per_class = [float(np.sum(y == c)) for c in class_labels]
        return Node(0, len(y), num_samples_per_class, class_labels)

    def predict(self, X):
        return [self.predict_sample(sample) for sample in X]

    def predict_sample(self, sample):
        node = self.root
        while not node.is_leaf_node():
            if sample[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_labels[np.argmax(node.num_samples_per_class)]


# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only take the first two features
y = (iris.target != 0).astype(int)  # Map to binary classification (Setosa: 0, Versicolor: 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Predict and evaluate the model
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Versicolor', 'Setosa'], yticklabels=['Versicolor', 'Setosa'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()