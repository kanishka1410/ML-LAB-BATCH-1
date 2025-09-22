
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ID3 uses 'entropy'
clf_id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf_id3.fit(X_train, y_train)

print("ID3 Accuracy:", clf_id3.score(X_test, y_test))

# Visualize
plt.figure(figsize=(12,8))
plot_tree(clf_id3, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
