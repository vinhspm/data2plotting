import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
from sklearn import datasets
from sklearn.manifold import TSNE
import random

def kmeans_display(x, label):
    k = np.amax(label) + 1
    x0 = x[label == 0, :]
    x1 = x[label == 1, :]
    x2 = x[label == 2, :]
    x3 = x[label == 3, :]
    x4 = x[label == 4, :]
    x5 = x[label == 5, :]
    x6 = x[label == 6, :]
    x7 = x[label == 7, :]
    x8 = x[label == 8, :]
    x9 = x[label == 9, :]
    
    plt.scatter(x0[:, 0], x0[:, 1],s=1,c = '#00AA55')
    plt.scatter(x1[:, 0], x1[:, 1],s=1,c = '#4183D7')
    plt.scatter(x2[:, 0], x2[:, 1],s=1,c = '#FF00FF')
    plt.scatter(x3[:, 0], x3[:, 1],s=1,c = '#050709')
    plt.scatter(x4[:, 0], x4[:, 1],s=1,c = '#AA8F00')
    plt.scatter(x5[:, 0], x5[:, 1],s=1,c = '#AA6B51')
    plt.scatter(x6[:, 0], x6[:, 1],s=1,c = '#E0FFFF')
    plt.scatter(x7[:, 0], x7[:, 1],s=1,c = '#00AAAA')
    plt.scatter(x8[:, 0], x8[:, 1],s=1,c = '#DCC6E0')
    plt.scatter(x9[:, 0], x9[:, 1],s=1,c = '#C9F227')

    plt.axis('equal')
    plt.show()

def kmeans_init_centers(x, k):
    # randomly pick k rows of X as initial centers
    return x[np.random.choice(range(x.shape[0]), k, replace=False), :]
def kmeans_assign_labels(x, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(x, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)
def kmeans_update_centers(x, labels, k):
    centers = np.zeros((k, x.shape[1]))
    for k in range(k):
        # collect all points assigned to the k-th cluster 
        xk = x[labels == k, :]
        # take average
        centers[k,:] = np.mean(xk, axis = 0)
    return centers
def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))
f = open('dataset2.txt',"r",encoding="utf-8")
imgs =f.readlines()
plotting = []
for i in range(0, len(imgs)):
    imgs[i] = imgs[i].split(' ')
for i in imgs:
    for j in range(0,len(i)):
        i[j] = int(i[j])

for i in range(len(imgs)):    
    imgs[i] = np.array(imgs[i])
    plotting.append(imgs[i])
    imgs[i] = imgs[i].reshape(28,28)
plotting = np.array(plotting)

def kmeans(x, k):
    centers = [kmeans_init_centers(x, k)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(x, centers[-1]))
        new_centers = kmeans_update_centers(x, labels[-1], k)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

plotting = TSNE(n_components=2, learning_rate=200).fit_transform(plotting)

(centers, labels, it) = kmeans(plotting, 10)
print(len(centers))
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(plotting, labels[-1])
