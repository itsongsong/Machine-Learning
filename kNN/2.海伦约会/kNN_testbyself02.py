import numpy as np
import operator

"""
KNN步骤：
1. 处理数据格式
2. 数据归一化
3. 计算欧式距离，求出分类值
"""
def test():
    # 标签
    labelData = {'didntLike': '讨厌', 'smallDoses': '一般', 'largeDoses': '喜欢'}

    # 三维数据
    gamePercent = float(input('游戏时间占比：'))
    flyTime = float(input("飞行时长："))
    iceCream = float(input("冰淇淋数："))

    # 处理数据
    filename = 'datingTestSet.txt'
    # 获得数据和标签
    testData, testLabel = fileToMatrix(filename)

    # 归一化
    listNormData, rangeData, minData = normData(testData)

    # 样本数据归一化
    sampleData = ([gamePercent, flyTime, iceCream] - minData) / rangeData

    # 验证数据
    verifyRes = verifyData(sampleData, listNormData, testLabel, 3)


    print('喜欢程度：%s' % labelData[verifyRes])

"""
    回归数据
    1. 求出样本数据到测试数据的欧式距离
    2. 从小到大排序，求出前key个最小值
    3. 返回前key个最小值中最多的标签数据
"""
def verifyData(sampleData, testData, labelData, key):

    lines = len(testData)
    listSampleData = np.tile(sampleData, (lines, 1))

    # 计算欧式距离
    oushiDistance = (np.sum((listSampleData - testData) ** 2, axis=1)) ** 0.5
    # 从小到达排序获取key值
    sortedData = oushiDistance.argsort()

    returnLabel = {}
    for i in range(key):
        # 当前标签名
        labelName = labelData[sortedData[i]]
        returnLabel[labelName] = returnLabel.get(labelName, 0) + 1

    # 返回标签最多的一个
    return sorted(returnLabel.items(), key=operator.itemgetter(1), reverse=True)[0][0]

"""
    数据归一化
    1. 求出最大值和最小值
    2. 计算范围值
    3. 求出每个数据的权重
"""
def normData(originData):
    minData = originData.min(0)
    maxData = originData.max(0)
    rangeData = maxData - minData

    lines = len(originData)
    return (originData - np.tile(minData, (lines, 1))) / np.tile(rangeData, (lines, 1)), rangeData, minData


"""
    从文件中读取数据，并处理成测试数据和标签
    1. 读取文件数据
    2. 将数据分类成测试数据和变迁数据
"""
def fileToMatrix(filename) :
    # 打开文件
    fr = open(filename)

    # 读取内容
    fileContent = fr.readlines()

    # 文件行数
    fileLines = len(fileContent)

    # 返回数据
    returnData = np.zeros((fileLines, 3))
    returnLabel = []

    for i in range(fileLines):
        # 去除
        fileLineData = fileContent[i].strip()
        # 按照空格分组
        listFileLineData = fileLineData.split('\t')

        returnData[i, :] = listFileLineData[0: 3]
        returnLabel.append(listFileLineData[3])

    return returnData, returnLabel


if __name__ == '__main__' :
    test()
