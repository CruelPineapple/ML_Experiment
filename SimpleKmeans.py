import numpy as np

class SimpleKmeans:
    def __init__(self, k, iter=200):
        self.k = k
        self.iter = iter

    def fit(self, x, centers=None):
        #  第一步,随机选择k个点,或者指定
        if centers is None:
            idx = np.random.randint(low=0, high=len(x), size=self.k)
            centers = x[idx]
        inters = 0
        previous_center = None
        while inters < self.iter:
            print("%d-th iteration" % inters)
            points_set = {key: [] for key in range(self.k)}  #  k个类
            #  第二步, 遍历所有点P, 将P放入最近的举类中心的集合中
            for p in x:
                distances = np.sum((centers - p)**2, axis=1)
                nearest_index = np.argmin(distances)
                points_set[nearest_index].append(p)
            #  第三步, 遍历每一个点集, 计算新的据类中心
            for i_k in range(self.k):
                centers[i_k] = sum(points_set[i_k]) / len(points_set[i_k])
            if inters < 1:
                inters += 1
                previous_center = centers
                continue
            dif = np.array(previous_center-centers)
            #  计算前后两次中心的距离, 如果距离很近,意味着分类相差甚微,结束
            if sum(np.linalg.norm(d) for d in dif) < 0.00001:
                break
            else:
                previous_center = centers
            inters += 1
        return points_set, centers


if __name__ == '__main__':
    data = [
        [0., 2.],
        [0., 0.],
        [1., 0.],
        [5., 0.],
        [5., 2.]
    ]
    m = SimpleKmeans(2, iter=5)
    dd = np.array(data)
    init_center = [
        [0., 0.],
        [0., 2.]
    ]
    cc = np.array(init_center)
    point_set, centers = m.fit(dd, centers=cc)
    print(point_set)
    print(centers)
