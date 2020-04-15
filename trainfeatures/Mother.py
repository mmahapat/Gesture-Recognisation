import pandas as pd
import numpy
from sklearn import svm
from pathlib import Path

counter = 0
minimum = 250
datasize = numpy.array([])

def TrainMother(featureVectorMother, featureMatrixMother):
    global counter

    if counter == 0:
        featureMatrixMother = numpy.concatenate([[featureVectorMother]])
    else:
        featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]), axis=0)

    counter = counter + 1

    return featureMatrixMother


def PreProcess(Filepath):
    global minimum
    global datasize

    rawData = pd.read_csv(
        Filepath,
        sep=',', header=None)

    rY = rawData[34]

    rY = numpy.array(rY.array[1:])
    rY = rY.astype(numpy.float)
    normRawData = (rY - numpy.mean(rY)) / (numpy.max(rY - numpy.mean(rY)) - numpy.min(rY - numpy.mean(rY)))

    diffNormRawData = numpy.diff(normRawData)
    minimum = min(len(diffNormRawData), minimum)
    datasize = numpy.append(datasize, len(diffNormRawData))
    for i in range(180 - len(diffNormRawData)):
        diffNormRawData = numpy.append(diffNormRawData, diffNormRawData[i])
    zeroCrossingArray = numpy.array([])
    maxDiffArray = numpy.array([])

    if diffNormRawData[0] > 0:
        initSign = 1
    else:
        initSign = 0

    windowSize = 5

    for x in range(1, len(diffNormRawData)):
        if diffNormRawData[x] > 0:
            newSign = 1
        else:
            newSign = 0

        if initSign != newSign:
            zeroCrossingArray = numpy.append(zeroCrossingArray, x)
            initSign = newSign
            maxIndex = numpy.minimum(len(diffNormRawData), x + windowSize)
            minIndex = numpy.maximum(0, x - windowSize)

            maxVal = numpy.amax(diffNormRawData[minIndex:maxIndex])
            minVal = numpy.amin(diffNormRawData[minIndex:maxIndex])

            maxDiffArray = numpy.append(maxDiffArray, (maxVal - minVal))

    index = numpy.argsort(-maxDiffArray)

    featureVectorMother = numpy.array([])
    featureVectorMother = numpy.append(featureVectorMother, diffNormRawData)
    featureVectorMother = numpy.append(featureVectorMother, zeroCrossingArray[index[0:5]])
    featureVectorMother = numpy.append(featureVectorMother, maxDiffArray[index[0:5]])

    for i in range(190 - len(featureVectorMother)):
        featureVectorMother = numpy.append(featureVectorMother, [0])
    return featureVectorMother


def main():
    global counter
    featureMatrixMother = numpy.array([])
    pathlistMother = Path('../traindata/').glob('**/*.csv')
    for path in pathlistMother:
        path_in_str = str(path)
        featureVectorMother = PreProcess(path_in_str)
        featureMatrixMother = TrainMother(featureVectorMother, featureMatrixMother)

    # counter = 0
    # featureMatrixNotMother = numpy.array([])
    # pathlistNotMother = Path('../notMother/').glob('**/*.csv')
    # for path in pathlistNotMother:
    #     path_in_str = str(path)
    #     featureVectorNotMother = PreProcess(path_in_str)
    #     if featureVectorNotMother.size != 190:
    #         print("Bad")
    #     featureMatrixNotMother = TrainMother(featureVectorNotMother, featureMatrixNotMother)



    featureMatrixNotMother = featureMatrixMother - numpy.random.rand(60, 190)
    TrainingSamples = numpy.concatenate((featureMatrixMother, featureMatrixNotMother), axis=0)

    labelVector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    clf = svm.SVC(probability=True)
    clf.fit(TrainingSamples, labelVector)
    pathlist_test = Path('../testdata/').glob('**/*.csv')
    for path in pathlist_test:
        path_in_str = str(path)
        TestVectorMother = PreProcess(path_in_str)
        TestMatrixMother = numpy.concatenate([[TestVectorMother]])
        print(path)
        print(clf.predict(TestMatrixMother))
        print(clf.predict_proba(TestMatrixMother))
        print("Okay")

    print("Finish")



if __name__ == "__main__":
    main()
