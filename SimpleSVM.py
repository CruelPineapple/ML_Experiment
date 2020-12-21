import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#  构造7.22式, 函数中的abcd分别为训练样本对应的系数,即alpha1-4

def creat(co, X, y, a, b, c, d, e):
    L_0 = co * X * y
    L_1 = L_0.sum(axis=0)
    L = np.dot(L_1, L_1) / 2 - co.sum()
    #  将e=a+b+c-d带入, 化简整理
    L = sp.expand(L.subs(e, a + b + c -d))
    return L


#  求解拉格朗日函数
def _find_min(L, num):
    #  L为拉格朗日函数, num为变量个数
    pro_res = []
    #  调用sp库计算拉格朗日函数的导数
    derivative_of_L = sp.diff(L, num)
    #  求导后等于0 求解方程,即得到最优点候选
    res = sp.solve(derivative_of_L, list(num))
    if res:  #  如果方程有解
        if _judge(res):  #  如果方程有唯一不小于0且不全为0的实数解
            pro_res.append(res)
            return pro_res
        else:  #  方程有无数组解, 到子边界寻找极值点
            value = _find_submin(L, num)
            pro_res.append(value)
    else:
        #  方程无解, 到子边界寻找极值点
        value = _find_submin(L, num)
        pro_res.append(value)
    return pro_res

def _judge(res):
    for s in res.values():
        try:
            if float(s) < 0:
                return False  #  有负值分量则不合要求
        except:
            return False
    #  这里不能用各分量之和来判定, 先排除分量有负值, 才能因和不为0肯定解符合要求
    return True if sum(res.values()) != 0 else False

#  若L求导后为0的这个方程无解, 则从L的多个边界求解
def _find_submin(L, num):
    if num.shape[0] == 1:
        return None
    else:
        res = []
        for i in range(num.shape[0]):
            #  对于每个训练样本, 尝试删掉他后求解优化问题, 如果有解则加入候选
            L_subset = L.subs({num[i]:0})
            num_subset = np.delete(num, i, axis=0)
            #  实现求解子问题, 如果还是无解则又进入本函数,递归删去训练样本
            res.append(_find_min(L_subset, num_subset))
        return res

def find_min(L, num, a, b, c, d, e):
    #  求解所有可能的极小值点
    results = _find_min(L, num)
    reset(results)
    L_min = float("inf")
    res = None
    #  在所有边界最小值中选取是的L(a,b,c,d)最小的点
    for i in res_list:
        d_i = dict()
        for j in [a, b, c, d]:
            d_i[j] = i.get(j, 0)
        result = L.subs(d_i)
        if result < L_min:
            L_min = result
            res = d_i
    #  将e计算出来并添加到res中
    res[e] = res[a] +res[b] + res[c] - res[d]
    return res

#  将结果取出, 存放在res_list里
def reset(res):
    if not isinstance(res[0], list):
        if res[0]:
            res_list.append(res[0])
    else:
        for i in res:
            reset(i)

#  计算w b
def calculate_w_b(X, y, res):
    alpha = np.array([[i] for i in res.values()])
    w = (alpha * X * y).sum(axis=0)
    for i in range(alpha.shape[0]):
        if alpha[i]:
            b = y[i] - w.dot(X[i])
            break
    return w, b

def main():
    #  构建目标函数L(a,b,c,d,e)
    a, b, c, d, e = sp.symbols("a,b,c,d,e")
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 2]
    ])
    y = np.array([[1],[1],[1],[-1],[-1]])
    co = np.array([[a],[b],[c],[d],[e]])
    L = creat(co, X, y, a, b, c, d, e)
    num = np.array([a, b, c, d])
    # 求解极小值点
    global res_list  #  这里声明全局变量
    res_list = []
    res = find_min(L, num, a, b, c, d, e)
    #  求w b
    w, b = calculate_w_b(X, y, res)
    print("w", w)
    print("b", b)
    #  绘制样本点,分离超平面和间隔边界
    yy = np.array([y[i][0] for i in range(y.shape[0])])
    X_positive = X[np.where(yy == 1)]  #  正例
    X_negative = X[np.where(yy == -1)]  #  负例
    x_1 = X_positive[:, 0]  #  横坐标
    y_1 = X_positive[:, 1]  #  纵坐标
    x_2 = X_negative[:, 0]
    y_2 = X_negative[:, 1]
    plt.plot(x_1, y_1, "ro")
    plt.plot(x_2, y_2, "gx")
    xxx = np.array([0, 3])
    yyy = (-b - w[0] * xxx) / w[1]
    y_positive = (1 - b -w[0] * xxx) / w[1]
    y_negative = (-1 - b -w[0] * xxx) / w[1]
    plt.plot(xxx, yyy, "r-")
    plt.plot(xxx, y_positive, "b-")
    plt.plot(xxx, y_negative, "b-")
    plt.show()


if __name__ == '__main__':
    main()