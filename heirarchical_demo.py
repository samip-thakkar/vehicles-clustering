# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import the libraries
import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 


"""Generating a random dataset, with 50 samples and 4 clusters with random centroids."""
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
#Scatter the points
plt.scatter(X1[:, 0], X1[:, 1], marker = 'o')

"""Agglomerative Clustering"""
#We will have two inputs: number of clusters = 4, linkage type = 'average'
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')

#Fit the model
agglom.fit(X1, y1)

"""Display the clustering."""

#Crate a figure
plt.figure(figsize = (10, 6))

#Scale the datapoints
x_min, x_max = np.min(X1, axis = 0), np.max(X1, axis = 0)

#Get the average of data
X1 = (X1- x_min) / (x_max - x_min)

#Display all the datapoints
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
#Remove the xticks and yticks
plt.xticks([])
plt.yticks([])
plt.axis('off')
#Display plot of data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker = '.')
#Display the plot
plt.show()    

"""Prepare the distance matrix."""
dist_matrx = distance_matrix(X1, X1)
print(dist_matrx)

#Use the complete linkage and distance matrix for creating hierarchy
Z = hierarchy.linkage(dist_matrx, 'complete')

#Create the dendrogram
dendro = hierarchy.dendrogram(Z)

"""Changing for average linkage"""

#Use the complete linkage and distance matrix for creating hierarchy
Z = hierarchy.linkage(dist_matrx, 'average')

#Create the dendrogram
dendro = hierarchy.dendrogram(Z)

"""Ask user the type of linkage"""
try:
    inp = input("Enter type of linkage: average, complete, single or centroid: ")
    Z = hierarchy.linkage(dist_matrx, inp)
    dendro = hierarchy.dendrogram(Z)
except:
    print("Enter correct linkage")