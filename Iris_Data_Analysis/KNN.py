from sklearn.datasets import load_iris
iris = load_iris()


X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

'''
KNN Algoritması ile eğitim

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

#lineer Regresyon ile eğitim

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Lineer regresyonda hata verir!!! 3 sınıflı sınıflandırma yapmaz.
'''

# SVM algoritması ile eğitim
from sklearn.svm import SVC
model = SVC()


model.fit(x_train, y_train)
y_tahmin = model.predict(x_test)
print(y_tahmin)


# Hata matrisi 
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(y_test, y_tahmin)
print(hata_matrisi)

# Hata matrisinin plot çizimi 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
index = ['setosa','versicolour','virginica']
columns = ['setosa','versicolour','virginica']
hata_goster = pd.DataFrame(hata_matrisi,columns,index)
plt.figure(figsize=(10,6))
sns.heatmap(hata_goster, annot=True)


