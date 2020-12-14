import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # 贝叶斯系数, 取0时, 即为极大似然估计, 取1为贝叶斯估计中的拉普拉斯平滑
        self.y_types_count = None  # y的(类型: 数量)
        self.y_types_proba = None  # y的(类型: 概率)
        self.x_types_proba = dict()  # (xi的编号, xi的取值,y的类型):概率

    def train(self, X_train, y_train):
        self.y_types = np.unique(y_train)  # y的所有取值类型
        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y_train)   # 转换成pandas DataFrame数据格式
        # y的(类型: 概率) 计算, 公式4.11
        self.y_types_count = y[0].value_counts()
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)

        # (xi的编号, xi的取值, y的类型):条件概率计算
        for idx in X.columns:  # 遍历xi
            for j in self.y_types:  # 选取每一个y的类型
                p_x_y = X[(y == j).values][idx].value_counts()  # 选择诉讼有y==j为真的数据点的第idx各特征的值,并对这些值进行(类型:数量)统计
                for i in p_x_y.index:   # 计算(xi的编号, xi的取值, y的类型):概率
                    self.x_types_proba[(idx, i, j)] = (p_x_y[i] + self.lambda_) / (
                        self.y_types_count[j] + p_x_y.shape[0] * self.lambda_
                    )

    def predict(self, X_new):
        res = []
        for y in self.y_types:  # 遍历y可能的取值
            p_y = self.y_types_proba[y]  # 计算y的鲜艳概率
            p_xy = 1
            for idx, x in enumerate(X_new):
                p_xy *= self.x_types_proba[(idx, x, y)]  # 计算p(X=(x1, x2 ... xd)/Y=ck)
            res.append(p_y * p_xy)
        for i in range(len(self.y_types)):
            print("[{}]对应概率: {:.2%}".format(self.y_types[i], res[i]))
        # 返回最大后验概率对应的y值
        return self.y_types[np.argmax(res)]

def main():
    X_train = np.array([
        [1, "S"],
        [1, "M"],
        [1, "M"],
        [1, "S"],
        [1, "S"],
        [2, "S"],
        [2, "M"],
        [2, "M"],
        [2, "L"],
        [2, "L"],
        [3, "L"],
        [3, "M"],
        [3, "M"],
        [3, "L"],
        [3, "L"],
    ])
    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    clf = NaiveBayes(lambda_=0.0)  # lambda_=1.0为拉普拉斯平滑
    clf.train(X_train, y_train)
    X_new = np.array([2, "S"])   # 课本p65例子
    y_predict = clf.predict(X_new)
    print("{}被分类为:{}".format(X_new, y_predict))  # 核对概率计算结果和分类结果


if __name__ == "__main__":
    main()
