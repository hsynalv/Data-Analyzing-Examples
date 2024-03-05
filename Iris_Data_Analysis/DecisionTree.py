from sklearn.datasets import load_iris
iris = load_iris()

'''
print(iris.feature_names) # veri seti özelliklerinin sahip olduğu kolon adı ve veri türünü verir
print(iris.target_names) # veri setinin çıkış özelliğni verir

print(iris.target) # veri setinin çıkış özelliğini sayısal olarak verir
print(iris.data) # veri setinin giriş değerlerini verir
'''

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("Eğitim veri seti boyutu:", len(x_train))
print("Test veri seti boyutu:", len(x_test))


# Karar ağacı sınıflandırıcısını çağırarak fit() fonksiyonu ile eğitimi 
#  gerçekleştiririz. predict() fonksiyonu ile de tahmin yürütürüz
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(x_train,y_train)
y_tahmin = model.predict(x_test)
print(y_tahmin)


# Hata matrisi 
# Çıktıda verilen matriste sayıların toplamı test veri seti boyutuna eşit olmak zorunda yoksa hata yapılmış demektir.
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