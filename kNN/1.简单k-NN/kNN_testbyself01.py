import numpy as np
import operator

"""
1. 创建数据集
2. 求测试点到数据集中所有点的距离并返回最小值
"""

def createDataCollect():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def judgeLabels(testData, collectData, labelsData, k):
    data = np.sum((testData - collectData) ** 2, 1) ** 0.5
    # 排序 返回键
    sortData = data.argsort()
    # 找到前k个键的标签
    returnLabel = {}
    for i in range(k):
        returnLabel[labelsData[sortData[i]]] = returnLabel.get(labelsData[sortData[i]], 0) + 1
    return sorted(returnLabel.items(), key=operator.itemgetter(1), reverse = True)[0][0]

if __name__ == '__main__':
    group, labels = createDataCollect()

    # 测试数据
    testData = [15, 115]

    # 分类
    label = judgeLabels(testData, group, labels, 3)

    print(label)



