import numpy as np
import pandas as pd
import time
from collections import namedtuple


class Node(namedtuple("Node", "children content feature feature_value label")):
    #  节点内容: 子节点, 节点内容, 节点分类特征,特征取值, 标签
    def __repr__(self):
        return str(tuple(self))


class DecisionTree:
    def __init__(self):
        self.tree = None

    def _gini(self, D):
        #  D, 待计算的集合
        #  return 该集合的基尼指数,按分布计算

        #  统计每个取值出现的频率
        numerator = D.iloc[:, 0].value_counts()
        x_type_prob = numerator / D.shape[0]
        #  计算基尼指数
        x_gini = 1 - sum(p * p for p in x_type_prob)
        return x_gini

    def _giniUnderFeature(self, X, y):
        #  X,feature A
        #  y, Class D
        #  return: Gini(D, A=?)中最佳者

        feature_values = np.unique(X)
        rst = []
        retval = []
        for feature_value in feature_values:
            yes_index = np.array((X == feature_value).values)
            D_1 = y[yes_index]
            D_2 = y[~yes_index]
            gini_vals = D_1.shape[0] / y.shape[0] * self._gini(D_1) + \
                        D_2.shape[0] / y.shape[0] * self._gini(D_2)
            rst.append(gini_vals)
        best_index = np.argmin(rst)
        retval.append(feature_values[best_index])
        retval.append(rst[best_index])
        return retval

    def _choose_feature(self, X_train, y_train, features):
        #  选择分类特征
        gini_val = []
        feature_val = []
        for feature in features:
            val = self._giniUnderFeature(X_train[feature], y_train)
            gini_val.append(val[1])
            feature_val.append(val[0])
        idx = np.argmin(gini_val)
        #  print("best gini: {}".format(gini_val[idx]]
        ret = [features[idx], feature_val[idx], gini_val[idx]]
        return ret

    def _built_tree(self, X_train, y_train, features):
        #  只有一个节点, 或已经完全分类, 则决策树停止继续分叉
        #  实际上还可以放宽到gini小于阈值(需求出优势标签),请自己实现
        if len(features) == 1 or len(np.unique(y_train))==1:
            label = list(y_train[0].value_counts().index)[0]
            return  Node(children=None, content=(X_train, y_train), feature=None,
                         feature_value=None, label=label)
        else:
            #  选择分类特征值
            feature_value = self._choose_feature(X_train, y_train, features)
            feature = feature_value[0]
            value = feature_value[1]
            features.remove(feature)
            #  构建节点, 同时递归创建孩子节点
            yes_idx = X_train[feature_value[0]] == feature_value[1]
            no_index = ~np.array(yes_idx)
            X_item_leftChild = X_train[yes_idx]
            X_item_rightChild = X_train[no_index]
            y_item_leftChild = y_train[yes_idx]
            y_item_rightChild = y_train[no_index]
            child = [self._built_tree(X_item_leftChild, y_item_leftChild, features),
                     self._built_tree(X_item_rightChild, y_item_rightChild, features)]
            return Node(children=child, content=None, feature=feature,
                        feature_value=value, label=None)

    def train(self, X_train, y_train, features):
        self.tree = self._built_tree(X_train, y_train, features)

    def _search(self, tree, X_new):
        if tree.feature_value is None:
            return tree.label
        if X_new[tree.feature].loc[0] == tree.feature_value:
            ret = self._search(tree.children[0], X_new)  #  0号元素总是为左节点
        else:
            ret = self._search(tree.children[1], X_new)  #  1号元素总是为右节点
        return ret

    def predict(self, X_new):
        tree = self.tree
        return self._search(tree, X_new)

def main():
    star = time.time()

    features = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    X_train = np.array([
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "否", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ])
    y_train = np.array(["否", "否", "是", "是", "否", "否", "否", "是", "是",
                        "是", "是", "是", "是", "是", "否"])

    X_train = pd.DataFrame(X_train, columns=features)
    y_train = pd.DataFrame(y_train)

    clf = DecisionTree()

    #  测试代码,请与课本84-85页的例子核对
    print(clf._gini(y_train))
    print("特征A下集合D的基尼指数")  #  请与课本例子核对
    print(clf._giniUnderFeature(X_train["年龄"], y_train))
    print("特征选择结果")  #  与课本例子核对
    print(clf._choose_feature(X_train, y_train, features))

    clf.train(X_train, y_train, features.copy())

    X_new = np.array([["青年", "是", "否", "一般"]])
    # X_new = np.array([["青年", "否", "否", "好"]])
    X_new = pd.DataFrame(X_new, columns=features)
     # print(X_new["年龄"].loc[0])
    y_predict = clf.predict(X_new)
    print("预测结果:{}".format(y_predict))
    print("time:{:.4f}s".format(time.time()-star))


if __name__ == '__main__':
    main()