from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
print(iris['data'][:,::2].shape)

data = iris['data'][:,:2]
x = data[:,0]
y = data[:,1]
plt.scatter(x, y, color='green')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()