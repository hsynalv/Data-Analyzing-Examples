import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []

kume_sayisi_listesi = range(1,11)
for i in kume_sayisi_listesi:
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(kume_sayisi_listesi, wcss)
plt.title("Küme Sayısı Belirlemek İçin Dirsek Yöntemi")
plt.xlabel("Küme Sayısı")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Küme 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Küme 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Küme 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Küme 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Küme 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c= 'yellow', label='Küme Merkezleri')
plt.title("Müşteri Segmentasyonu")
plt.xlabel("Yıllık Gelir")
plt.ylabel("Yıllık Harcama Oranı")
plt.show()