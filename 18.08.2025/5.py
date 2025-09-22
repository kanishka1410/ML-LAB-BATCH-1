pip install decision-tree-id3

from id3 import Id3Estimator, export_graphviz
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train using ID3 (package internally handles Gain Ratio ~ C4.5 style)
estimator = Id3Estimator(gain_ratio=True)  # C4.5 = Gain Ratio
estimator.fit(X, y)

print("C4.5 Rules:\n", estimator.tree_)
