import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def Functions(testSet, predictions):
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    for i in range(0, len(testSet)):
                if predictions[i] == '2':
                        if predictions[i] == testSet[i][-1]:
                                true_positives=true_positives+ 1
                        else:
                                false_positives=false_positives+1
                else:
                        if predictions[i] == testSet[i][-1]:
                                true_negatives =true_negatives+ 1
                        else:
                                false_negatives =false_negatives+ 1
    print('\n')
    print('\n') 
    print('\n')                           
    print("True Positive: ",true_positives)
    print("False Negative: ",false_negatives)
    print("True Negative: ",true_negatives)
    print("False Positive: ",false_positives)
    print('\n')
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
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset('haberman.data', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = Functions(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
main()
