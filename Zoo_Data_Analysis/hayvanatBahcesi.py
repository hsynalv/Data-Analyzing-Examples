# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

veri = pd.read_csv("/Gün 4/hayvanatbahcesi.csv", encoding='ISO-8859-1')
veri = pd.get_dummies(veri, columns=['hayvan adi'])


X = veri.drop('sinifi', axis=1)
y = veri['sinifi']

X_egitim, X_test, y_egitim, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_egitim, y_egitim)


y_pred = model.predict(X_test)
dogruluk = metrics.accuracy_score(Y_test, y_pred)
f1_skor = metrics.f1_score(Y_test, y_pred, average="weighted",  zero_division=1)
precision = metrics.precision_score(Y_test, y_pred, average="weighted",  zero_division=1)
recall = metrics.recall_score(Y_test, y_pred, average="weighted",  zero_division=1)
print("----------------------------------------")
print("Modelin doğruluk oranı:", dogruluk)
print("F1-Skoru:", f1_skor)
print("Precision:", precision)
print("Recall:", recall)
print("----------------------------------------")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# k-NN (k Nearest Neighbors) modeli
knn_model = KNeighborsClassifier()
knn_model.fit(X_egitim, y_egitim)
knn_tahmin = knn_model.predict(X_test)
knn_accuracy = accuracy_score(Y_test, knn_tahmin)
print("k-NN Accuracy:", knn_accuracy)
print("----------------------------------------")

# Support Vector Machine (SVM) modeli
svm_model = SVC()
svm_model.fit(X_egitim, y_egitim)
svm_tahmin = svm_model.predict(X_test)
svm_accuracy = accuracy_score(Y_test, svm_tahmin)
print("SVM Accuracy:", svm_accuracy)
print("----------------------------------------")

# Decision Tree modeli
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_egitim, y_egitim)
decision_tree_tahmin = decision_tree_model.predict(X_test)
decision_tree_accuracy = accuracy_score(Y_test, decision_tree_tahmin)
print("Decision Tree Accuracy:", decision_tree_accuracy)


