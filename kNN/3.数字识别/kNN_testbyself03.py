import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def fileToVector(filename):
    # 打开文件
    fr = open(filename)
    fileContent = fr.readlines()

    returnContent = np.zeros((1, 1024))
    for i in range(len(fileContent)):
        # 如果不用int类型会出现一个警告
        # FutureWarning: Beginning in version 0.22, arrays of bytes/strings will be converted to decimal numbers
        # if dtype='numeric'. It is recommended that you convert the array to a float dtype before using it in
        # scikit-learn, for example by using your_array = your_array.astype(np.float64).  FutureWarning)
        for j in range(len(fileContent[i].strip())):
            returnContent[0, 32 * i + j] = int(fileContent[i][j])

    # ValueError: Expected 2D array, got 1D array instead:
    return returnContent


"""
利用sklearn来进行KNN运算：
1. 处理数据：获取文件，从文件中读取数据，并处理格式
2. 处理训练数据
3. 处理测试数据
4. 创建KNN分类器
"""
def test():
    trainingFile = 'trainingDigits'
    testFile = 'testDigits'
    # 获取文件夹下面的所有文件
    listFiles = listdir(trainingFile)
    # 文件个数
    numberOfFile = len(listFiles)
    # 标签
    listLabels = []
    # 文件内容
    listFileContents = np.zeros((numberOfFile, 1024))

    # 获取所有文件中的内容存储在一个list中
    for i in range(numberOfFile):
        # 获取文件名称
        fileName = listFiles[i]
        # 获取文件实际数字
        listLabels.append(int(fileName.split('_')[0]))
        # 处理文件内容
        listFileContents[i] = fileToVector('%s/%s' % (trainingFile, fileName))
    # 创建KNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合数据(测试数据和标签数据）
    neigh.fit(listFileContents, listLabels)

    # 获取测试数据文件
    listTestFiles = listdir(testFile)

    errorCount = 0
    for i in range(len(listTestFiles)):
        # 文件名称
        fileName = listTestFiles[i]
        # 图片数字
        testNumber = int(fileName.split('_')[0])
        # 获取测试向量
        testData = fileToVector('%s/%s' % (testFile, fileName))
        # 获取预测结果
        result = neigh.predict(testData)

        if(result[0] != testNumber):
            errorCount += 1
            print('测试结果：%d 真是结果：%d' % (result[0], testNumber))

    print('错误个数：%d，总数：%d，错误率：%f%%' % (errorCount, len(listTestFiles), 100 * errorCount / len(listTestFiles)))

if __name__ == '__main__':
    test()
