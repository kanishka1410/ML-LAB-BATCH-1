import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from google.colab import files
import pandas as pd
uploaded = files.upload()
df = pd.read_csv("Multiclass Diabetes Dataset.csv")
print(df.columns)
X=df[['Gender','AGE','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI',]]
y=df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9, 11]
results = {"K": [], "Uniform": [], "Distance": []}
for k in k_values:

    knn_uniform = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform"
    )
    knn_uniform.fit(X_train, y_train)
    y_pred_uniform = knn_uniform.predict(X_test)
    acc_uniform = accuracy_score(y_test, y_pred_uniform)
    knn_distance = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )
    knn_distance.fit(X_train, y_train)
    y_pred_distance = knn_distance.predict(X_test)
    acc_distance = accuracy_score(y_test, y_pred_distance)

    results["K"].append(k)
    results["Uniform"].append(acc_uniform)
    results["Distance"].append(acc_distance)

df_results = pd.DataFrame(results)
print("KNN Performance on Iris Dataset:\n")
print(df_results)

plt.figure(figsize=(8, 5))
plt.plot(df_results["K"], df_results["Uniform"], marker="o", label="Uniform Weights")
plt.plot(df_results["K"], df_results["Distance"], marker="s", label="Distance Weights")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN Classification Accuracy on Iris Dataset")
plt.legend()
plt.show()
