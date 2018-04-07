import numpy as np
import operator
from os import listdir


def classify(inX, dataSet, label, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqdiffMat = diffMat ** 2
    sqDistance = sqdiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistance = distance.argsort()
    classDict = {}
    for i in range(k):
        voteLabel = label[sortedDistance[i]]
        classDict[voteLabel] = classDict.get(voteLabel, 0) + 1
        sortedClassDict = sorted(classDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassDict[0][0]


def img2vector(fileName):
    returnVector = np.zeros((1,1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32*i+j] = int(lineStr[j])
    return returnVector


def handwritingClassTest():
    hwLabel = []
    trainingFileList = listdir('trainingDigits')
    # print("\n trainingFileList: "  + str(trainingFileList))
    # print len(trainingFileList)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        trainingFileStr = fileNameStr.split('.')[0]
        classNameStr = int(trainingFileStr.split('_')[0])
        hwLabel.append(classNameStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    # print("\n hwLabel: \n" + str(np.array(hwLabel)))


    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        testFileStr = fileNameStr.split('.')[0]
        classNameStr = int(testFileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classResult = classify(vectorUnderTest, trainingMat, hwLabel, 3)

        print("the classifier came back with " + str(classResult) + "  the real answer is " + str(classNameStr))

        if (classResult != classNameStr):
            errorCount += 1.0

    print("\n total number of error is: " + str(errorCount))
    print("\n recognition rate is: " + str((1 - (errorCount/float(mTest)))*100) + "%")


if __name__ == '__main__':
    handwritingClassTest()