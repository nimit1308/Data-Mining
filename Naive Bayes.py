import csv
import random
import math

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def Details(testSet, predictions):
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    for i in range(0, len(testSet)):
                if predictions[i] == 2:
                        if predictions[i] == testSet[i][-1]:
                                true_positives =true_positives+ 1
                        else:
                                false_positives =false_positives+ 1
                else:
                        if predictions[i] == testSet[i][-1]:
                                true_negatives =true_negatives+ 1
                        else:
                                false_negatives =false_negatives+ 1
    print('\n')
    print('\n')                            
    print('\n')
    print 'True Positive: ',true_positives
    print 'False Negative: ',false_negatives
    print 'True Negative: ',true_negatives
    print 'False Positive: ',false_positives

    print('\n')
    precision = float(true_positives) /(float(true_positives + false_positives))
    recall = float(true_positives) /float(true_positives + false_negatives)
    sensitivity = float(true_positives) / float(true_positives + false_negatives)
    specificity = float(true_positives) / float(true_positives + false_negatives)
    f1_score = 2.0 / ((1.0 / precision) + (1.0 / recall))
    total = true_positives+true_negatives+false_positives+false_negatives
    error_rate = float(false_positives+false_negatives)/float(total)
    print 'Precision: ',precision
    print 'Recall: ',recall
    print 'Sensitivity: ',sensitivity
    print 'Specificity: ',specificity
    print 'F1 score: ', f1_score
    print 'Error rate: ',error_rate
    return (correct/float(len(testSet))) * 100.0

def main():
    filename = 'haberman.data'
    splitRatio = 0.9
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    accuracy = Details(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)

main()
