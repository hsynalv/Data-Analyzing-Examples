print("KNN")
import numpy as np
import pandas as pd

train = pd.read_csv("mitbih_train.csv")
X_train = np.array(train)[:,:187]
Y_train = np.array(train)[:,187]

test = pd.read_csv("mitbih_test.csv")
X_test = np.array(test)[:,:187]
Y_test = np.array(test)[:,187]

from sklearn.neighbors import KNeighborsClassifier
gnb = KNeighborsClassifier()
gnb.fit(X_train, Y_train)

y_pred = gnb.predict(X_test)


#Çıktıyı çizdirme
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, y_pred)

index = ['No','S','V','F','Q']
columns = ['No','S','V','F','Q']

cm_df =pd.DataFrame(cm,columns,index)
plt.figure(figsize=(10,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")

from sklearn import metrics
print("Ac:", metrics.accuracy_score(Y_test,y_pred))
print("Precision:", metrics.precision_score(Y_test,y_pred, average='weighted'))
print("recall:", metrics.recall_score(Y_test,y_pred, average='weighted'))
print("F1 Score:", metrics.f1_score(Y_test,y_pred, average='weighted'))

