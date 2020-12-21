import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#  KNN分类算法函数定义,这里没有采用KD树,而是用遍历的方法寻找近邻点
def KNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  #shape[0]表示行数
    #  step1 计算距离,这里tile(A, reps):构造一个矩阵,通过A重复reps次得到
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  #  按元素求差值
    squareDiff = diff ** 2  #  将差值平方
    squareDist = np.sum(squareDiff, axis=1)  #  按行累加
    distance = squareDist ** 0.5  #  将差值平方和求开方
    #  step2 对距离排序, 这里 argsort() 返回排序后的索引值
    sortedDistIndices = np.argsort(distance)
    # print("shape of sortedDistIndices:", shape(sortedDistIndices))
    classCount = {}  #  定义一个字典

    for i in range(k):
        #  step3 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]
        tempVote = int(voteLabel)
        #  step4 计算k个最近邻中各类别出现的次数
        classCount[tempVote] = classCount.get(tempVote, 0) + 1
    #  step5 返回出现次数最多的类别标签
    maxCount = 0
    manIndex = -1
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return  maxIndex


def main():
    data = pd.read_csv("creditcard.csv")
    data = data.drop(['Time', 'Amount'], axis=1)
    #  得到class == 1 的数量
    number_one = len(data[data['Class'] == 1])
    #  class == 1 的索引
    number_one_index = np.array(data[data['Class'] == 1].index)
    #  class == 0 index
    number_zero_index = data[data['Class'] == 0].index
    #  随机选取和Class==1一样数量的         要选择的列       要选择的数量   是否有放回抽样
    random_zero_index = np.random.choice(number_zero_index, number_one, replace=False)
    random_zero_index = np.array(random_zero_index)
    #  拼接数组
    sample = np.concatenate([random_zero_index, number_one_index])
    sample_data = data.iloc[sample, :]  #  按索引获取行


    x_sample_data = sample_data.iloc[:, sample_data.columns != 'Class']  #  属性值部分
    y_sample_data = sample_data.iloc[:, sample_data.columns == 'Class']  #  类别标签部分
    x_train_sample, x_test_sample, y_train_sample, y_test_sample = \
        train_test_split(x_sample_data, y_sample_data, test_size=0.3, random_state=0)

    k = 5
    rightSample = 0
    for i in range(len(x_test_sample)):
        testX = x_test_sample.values[i]
        outputLabel = KNNClassify(testX, np.array(x_train_sample), np.array(y_train_sample), k)
        print("Your input is:", testX, "and classified to class =>", outputLabel)
        if outputLabel == y_test_sample.values[i]:
            rightSample += 1

    print("正确率为:", rightSample / len(x_test_sample))


if __name__ == '__main__':
    main()

