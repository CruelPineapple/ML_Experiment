from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris['data'][:, :2].shape)
data = iris['data'][:, :2]

# kmeans = KMeans(n_clusters=3, max_iter=100).fit(data)
# gt_labels__ = kmeans.labels_
# centers__ = kmeans.cluster_centers_
#
# cat1 = data[gt_labels__ == 0]
# cat2 = data[gt_labels__ == 1]
# cat3 = data[gt_labels__ == 2]
#
# for ix, p in enumerate(centers__):
#     plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^', edgecolors='black', s=256)
#
# plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
# plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
# plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
# plt.title('Kmeans using sklearn with k=3')
# plt.xlim(4, 8)
# plt.ylim(1, 5)
# plt.show()


loss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=100).fit(data)
    loss.append(kmeans.inertia_ / len(data) / 3)
    #  这里是样本到聚类中心的最近距离的和,对样本取平均, 越小越好

plt.title('K with loss')
plt.plot(range(1, 10), loss)
plt.show()