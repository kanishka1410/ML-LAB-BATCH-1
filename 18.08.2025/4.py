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

# CART uses 'gini'
clf_cart = DecisionTreeClassifier(criterion="gini", random_state=42)
clf_cart.fit(X_train, y_train)

print("CART Accuracy:", clf_cart.score(X_test, y_test))

# Visualize
plt.figure(figsize=(12,8))
plot_tree(clf_cart, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
