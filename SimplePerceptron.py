import numpy as np
import time

def loadData(fileName, type=None):
    #  加载Mnist数据集, fileName:要加载的数据集路径
    #  返回: list形式的数据集及标记

    print('start to read data')

    dataArray = []
    labelArray = []
    i = 0

    fr = open(fileName, 'r')
    for line in fr.readlines():
        currentLine = line.strip().split(',')
        #  Mnist有0-9个标记, 由于是二分类任务, 所以将>=5的作为1, <5作为-1
        if int(currentLine[0])>=5:
            labelArray.append(1)
        else:
            labelArray.append(-1)
        dataArray.append([int(num) / 255 for num in currentLine[1:]])
        # 每一行的数据,除第一列是标记为,其余是训练样本属性
        # 将属性值除255归一化(非必要步骤
        i += 1
        print("add %d-th %s sample" % (i, type))
    return dataArray, labelArray

def perceptron_train(dataArray, labenArray, iter=50):
    #  感知机训练过程
    #  dataArray:训练集的数据(list),labelArray: 训练集的标签(list)
    #  iter: 迭代次数, 默认50 return 训练好的w,b
    print('start to train')
    dataMat = np.mat(dataArray)
    #  将标签转换为矩阵后转置
    #  转置原因, 运算中需要单独取label中的某个元素, 如果是1xN矩阵,
    #  无法用label[i]读取
    labelMat = np.mat(labenArray).T
    #  获取数据矩阵的大小, 为m*n
    m, n = np.shape(dataMat)
    #  创建初始权重w, 初始值全为0
    w = np.zeros((1, n))
    #  创建一个1行n列向量
    #  初始化偏置b为0
    b = 0
    #  初始化步长,控制梯度下降速率
    eta = 0.0001
    for k in range(iter):
        #  对于每一个样本进行梯度下降
        #  课本中在2.3.1的2.6式以前,使用的梯度下降,是全部样本都算一遍之后,
        #  进行一次梯度下降, 在公式2.6及算法2.1中, 用了随机梯度下降, 即
        # 计算一个样本就对该样本进行一次梯度下降.一般常用随机梯度下降
        for i in range(m):
            xi = dataMat[i]  #  获取当前样本的向量
            yi = labelMat[i]  # 获取当前样本对应的标签
            #  判断是否是误分类样本
            #  误分类样本特征为 -yi(w*xi+b)>=0
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + eta * yi * xi
                b = b + eta * yi
        print('Round %d of %d training' % (k, iter))
    return w, b


def model_test(testDataArray, labelArray, w, b):
    #  输入测试数据及其标签, 训练得到的wb
    print('start to test')
    dataMat = np.mat(testDataArray)
    labelMat = np.mat(labelArray).T
    m, n = np.shape(dataMat)
    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T +b)
        if result >= 0:
            errorCnt +=1
    accuRate = 1- (errorCnt / m)
    return accuRate

if __name__ == '__main__':
    start = time.time()
    print("load training data")
    trainData, trainLabel = loadData('mnist_train.csv', 'training')
    print("load test data")
    testData, testLabel = loadData('mnist_test.csv','test')

    w, b = perceptron_train(trainData, trainLabel, iter=30)
    accuRate = model_test(testData, testLabel, w, b)

    end = time.time()
    print('accuracy rate:', accuRate)
    print('time:', end - start)