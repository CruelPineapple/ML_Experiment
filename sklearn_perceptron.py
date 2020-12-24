import sklearn.linear_model as lm
import numpy as np
import SimplePerceptron as sp

print("load training data")
trainData, trainLabel = sp.loadData('mnist_train.csv', 'training')
print("load test data")
testData, testLabel = sp.loadData('mnist_test.csv', 'test')

perceptron = lm.Perceptron()
perceptron.fit(trainData, trainLabel)
w = perceptron.coef_
b = perceptron.intercept_
print("w:",w,"\n", "b:", b, "\n", "n_iter:", perceptron.n_iter_)

res = perceptron.score(trainData, trainLabel)
print("correct rate on training set:{:.0%}".format(res))

res2 = perceptron.score(testData, testLabel)
print("correct rate on test set:{:.0%}".format(res2))