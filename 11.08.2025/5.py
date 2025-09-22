from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L1 regularization
model_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
model_l1.fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)
print("L1 Regularization Classification Report:")
print(classification_report(y_test, y_pred_l1))

#L2 regularization
model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)
model_l2.fit(X_train, y_train)
y_pred_l2 = model_l2.predict(X_test)
print("L2 Regularization Classification Report:")
print(classification_report(y_test, y_pred_l2))

#ElasticNet regularization
model_en = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, l1_ratio=0.5)
model_en.fit(X_train, y_train)
y_pred_en = model_en.predict(X_test)
print("ElasticNet Regularization Classification Report:")
print(classification_report(y_test, y_pred_en))

