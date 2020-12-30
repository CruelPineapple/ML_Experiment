from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import SimpleKmeans as skms

#  读入iris数据集
iris = datasets.load_iris()
print(iris['data'][:, :2].shape)
data = iris['data'][:, :2]
m = skms.SimpleKmeans(3)
init_center = [
    [3.5, 1.5],
    [5.8, 1.5],
    [7, 1.5]
]
cc = np.array(init_center)
points_set, centers = m.fit(np.array(data), centers=cc)
print(points_set)
print(centers)
cat1 = np.asarray(points_set[0])
cat2 = np.asarray(points_set[1])
cat3 = np.asarray(points_set[2])

for ix, p in enumerate(centers):
    plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^', edgecolor='black', s=256)

plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
plt.title('Kmeans with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()