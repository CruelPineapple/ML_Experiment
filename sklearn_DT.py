from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time

from sklearn import tree
import pydotplus
import graphviz


def show(clf, features, y_types):
    #  决策树的可视化
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features,
                                    class_names=y_types,
                                    filled=True,rounded=True,
                                    special_characters=True)
    pydotplus.graph_from_dot_data(dot_data)
    gra = graphviz.Source(dot_data)
    gra.view()
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_jpg('DT_show.jpg')

def main():
    star = time.time()
    features = ["age", "work", "house", "credit"]
    X_train = pd.DataFrame([
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]])
    y_train = pd.DataFrame(["否", "否", "是", "是", "否", "否", "否", "是", "是",
                            "是", "是", "是", "是", "是", "否"])
    #  数据预处理
    le_x = preprocessing.LabelEncoder()
    le_x.fit(np.unique(X_train))
    X_train = X_train.apply(le_x.transform)
    le_y =  preprocessing.LabelEncoder()
    le_y.fit(np.unique(y_train))
    y_train = y_train.apply(le_y.transform)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    X_new = pd.DataFrame([["青年", "否", "是", "一般"]])
    X = X_new.apply(le_x.transform)
    y_predict = clf.predict(X)
    X_show = [{features[i]: X_new.values[0][i]} for i in range(len(features))]
    print("{0}被分类为:{1}".format(X_show, le_y.inverse_transform(y_predict)))
    print("time:{:.4f}s".format(time.time()-star))

    show(clf, features, [str(k) for k in np.unique(y_train)])

if __name__== "__main__":
    main()