from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_data = np.loadtxt('train_adult_pro.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('test_adult_pro.csv', delimiter=',', skiprows=1)

X_train, y_train = train_data[:,:-1],train_data[:,-1]
X_test, y_test = test_data[:,:-1],test_data[:,-1]

clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"sklearn test on train data: {train_accuracy}")
print(f"sklearn test on test data: {test_accuracy}")
