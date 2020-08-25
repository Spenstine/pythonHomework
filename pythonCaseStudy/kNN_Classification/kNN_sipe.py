"""
kNN -> k Nearest Neighbours
k -> arbiterary integer
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


def getDistance(p1, p2):
    '''
    p1 and p2 -> tuple(x, y) to represent point on curtasian plane
    output -> shortest distance between the two points
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(sum(np.square((p1 - p2))))

def getHighestCount(anyArray):
    '''
    return the random choice from a list of mode of anyArray
    '''
    returnList = []
    valDict = Counter(anyArray)
    maxVal = max(valDict.values())
    for key, value in valDict.items():
        if value == maxVal:
            returnList.append(key)
    return np.random.choice(returnList)

def getHighestCount_short(anyArray):
    '''
    return the first occuring mode of anyArray
    '''
    mode, count = ss.mode(anyArray) #ss.mstat.mode is also ok
    return mode

def find_k_neighbours(p, points, k = 5):
    '''
    return the k nearest neighbours of p
    '''
    points = np.array(points)
    distance = []
    for point in points:
        distance.append(getDistance(point, p))
    ind = np.argsort(distance)
    return ind[:k]

def kNN_predict(p, points, classes, k = 5):
    '''
    classify a point based on its location in grid
    '''
    assert len(classes) == len(points), 'points should correspond their class'
    classes = np.array(classes) #increase computation capabilities by converting to ndarray
    ind = find_k_neighbours(p, points, k)
    values = classes[ind]
    return getHighestCount(values)

def generateSyntheticData(n = 50):
    '''
    generates 2 * n points representing two different class
    the data generated is therefore bivariate
    the first n points belong to the first class
    the second n points belong to the second class
    '''
    #zeroMean is class of points with mean of 0
    #oneMean is class of points with mean of 1
    points = np.concatenate((ss.norm(0,1).rvs((n, 2)), ss.norm(1,1).rvs((n, 2))), axis = 0)
    classes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return points, classes

def classify_one_point(point = (0, 0), n = 50, k = 5):
    '''
    this is kind of the main program
    bivariate data is generated
    the point supplied as arguments is classified and its class printed
    a visualization of the data and the point is made
    n -> number of points in each class
    k -> integer for the number of nearest neighbours
    '''
    points, classes = generateSyntheticData(n)
    point_class = kNN_predict(point, points, classes, k)
    print(point_class)

    plt.figure('kNN')
    plt.clf() #clear already existing plot in frame
    plt.title('kNN visualization')
    #classes are two: zeroMean and oneMean
    #zeroMean class is red and represented by the first n points
    plt.plot(points[:n, 0], points[:n, 1], 'ro', label = 'zeroMean')
    
    #oneMean class is blue and represented by the last n points
    plt.plot(points[-n:, 0], points[-n:, 1], 'bo', label = 'oneMean')

    #point is green
    plt.plot(point[0], point[1], 'go', label = 'point')
    plt.show()
    print('green point belongs to red category') if point_class == 0 else print('green point belongs to blue category')

def make_prediction_grid(limits, h, points, classes, k):
    '''
    classify each point in the prediction grid
    '''
    x_min, x_max, y_min, y_max = limits
    xVals = np.arange(x_min, x_max, h)
    yVals = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xVals, yVals)
    prediction_grid = np.zeros_like(xx, dtype = int)
    for i, x in enumerate(xVals):
        for j, y in enumerate(yVals):
            point = np.array([x,y])
            prediction_grid[j, i] = kNN_predict(point, points, classes, k)
    return xx, yy, prediction_grid

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(points[:,0], points [:,1], c = classes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

'''dummy dataset generated randomly'''
#classify_one_point(point = (1, 1), k = 3)
#points, classes = generateSyntheticData(50)
#limits = (-3, 4, -3, 4)
#xx, yy, prediction_grid = make_prediction_grid(limits, 0.1, points, classes, 5)
#plot_prediction_grid(xx, yy, prediction_grid, 'kNN.png')
#open kNN.png to view the plot

'''real dataset from sciKit learn'''
iris = datasets.load_iris()
points = iris.data[:, 0:2]; classes = iris.target
filename = 'iris_kNN_5.png'
limits = (4, 8.5, 2, 4.5)
h = 0.1
k = 5
xx, yy, prediction_grid = make_prediction_grid(limits, h, points, classes, k)
#plot_prediction_grid(xx, yy, prediction_grid, filename) 

'''comparison of kNN_classifier from sklearn and my home-made classifier'''
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(points, classes)
sk_predictions = knn.predict(points)
my_predictions = np.array([kNN_predict(p, points, classes, k) for p in points])
print()
print('Consistency: ')
print('=' * len('consistency:'))
print()
print('my_prediction and sk_prediction: ' + str(np.round(100 * np.mean(my_predictions == sk_predictions), 2)) + '%')
print()
print('my_prediction and classes/outcomes: ' + str(np.round(100 * np.mean(my_predictions == classes), 2)) + '%')
print()
print('sk_prediction and classes/outcomes: ' + str(np.round(100 * np.mean(sk_predictions == classes), 2)) + '%')