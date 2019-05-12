import numpy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
import sys

from scipy.misc import imread

# data preperation (loading, normalizing, reshaping)
from init_centroids import init_centroids

#for given pixel, return the closest centroid
def classify(pixel,centroids):
    minDist = sys.float_info.max    #set max float value
    chosenCent = -1
    for idx, centroid in enumerate(centroids):  #iterate over each centroid(and keep its index)
        newDist = numpy.linalg.norm(pixel - centroid)
        if minDist > newDist:
            minDist = newDist
            chosenCent = idx
    return chosenCent

#with given number of iteration and the K centroids, print the values
def printIter(iterNum, centroids):
    print("iter:", iterNum, end=" ")
    for idx, centroid in enumerate(centroids):
        val = numpy.floor(centroids[idx] * 100) / 100
        print(val, end=" ")
    print("")

def KMeans(pic, k):
    print("k = ", k)
    centroids = init_centroids(pic, k)
    printIter(0, centroids)
    for i in range(1, 11):  # loop 10 times
        clusters = [[] for c in range(0, k)]    #create K clusters
        for pixel in pic:
            centIdx = classify(pixel, centroids)
            clusters[centIdx].append(pixel)
        #calculate average of cluster for new centroid value
        for idx,cluster in enumerate(clusters):
            sum = 0
            for pixel in cluster:   #sum pixel values in each cluster
                sum += pixel
            lenC = len(cluster)
            if lenC != 0:
                avg = sum / lenC
                centroids[idx] = avg
        printIter(i, centroids)

k=2
while (k <= 16):    #Run the Algorithm with k=2,k=4,k=8,k=16
    path = 'dog.jpeg'
    A = imageio.imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    KMeans(X, k)
    k*=2


