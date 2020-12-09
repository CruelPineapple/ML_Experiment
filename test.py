import pandas as pd
import numpy as np

y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
y = pd.DataFrame(y_train)
print(y)

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
X = pd.DataFrame(X_train)
print(X)

print(y[0].value_counts())
print(X[1].value_counts())

print(y.shape)
print(y.shape[0])
print(X.shape)
print(X.shape[1])

print(np.unique(y_train))

print(X[(y == 1)])
print(X[(y == 1).values])
print(X[(y == 1).values][0])
