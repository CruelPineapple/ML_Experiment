from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 2]
    ])
    y = np.array([1, 1, 1, -1, -1])
    clf = SVC(C=0.5, kernel="linear")
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_
    print(clf.support_vectors_)
    print(w, b)
    print(clf.predict([[5, 6], [-1, -1]]))
    print(clf.score(X, y))
    #  绘制
    yy = np.array([y[i] for i in range(y.shape[0])])
    X_positive = X[np.where(yy == 1)]
    X_negative = X[np.where(yy == -1)]
    x_1 = X_positive[:, 0]
    y_1 = X_positive[:, 1]
    x_2 = X_negative[:, 0]
    y_2 = X_negative[:, 1]
    plt.plot(x_1, y_1, "ro")
    plt.plot(x_2, y_2, "gx")
    xxx = np.array([0, 3])
    yyy = (-b - w[0] * xxx) / w[1]
    y_positive = (1 - b - w[0] * xxx) / w[1]
    y_negative = (-1 - b - w[0] * xxx) / w[1]
    plt.plot(xxx, yyy, "r-")
    plt.plot(xxx, y_positive, "b-")
    plt.plot(xxx, y_negative, "b-")
    plt.show()


if __name__ == '__main__':
    main()