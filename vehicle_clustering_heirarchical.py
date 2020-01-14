# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import the libraries
import pandas as pd
from scipy.cluster import hierarchy

#Read the data
data = pd.read_csv('cars.csv')

"""Data Cleaning."""
#Dropping the rows with null values
print("Shape before data cleaning: ", data.shape)
#Replacing null values with None
data[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = data[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
#Drop the null values
data = data.dropna()
data = data.reset_index(drop = True)
print("Shape after data cleaning: ", data.shape)

"""Feature Selection"""
featureset = data[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

"""Normalization"""
#Normalizing every values between 0 and 1, so that each value gets importance
from sklearn.preprocessing import MinMaxScaler
x = featureset.values
mms = MinMaxScaler()
feature_mtrx = mms.fit_transform(x)

"""Clustering using Scipy"""
import scipy
#Calculating the distance matrix
l = feature_mtrx.shape[0]
D = scipy.zeros([l, l])
for i in range(l):
    for j in range(l):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtrx[i], feature_mtrx[j])
#Creating linkage
Z = hierarchy.linkage(D, 'complete')

#Use cutting line in case of flat clustering while partitioning disjoint clusters
#Clustering for maximum depth
from scipy.cluster.hierarchy import fcluster
max_depth = int(input("Enter maximum depth of cluster: "))
clusters = fcluster(Z, max_depth, criterion='distance')
print(clusters)
#Use fclusters for maximum number of clusters
from scipy.cluster.hierarchy import fcluster
k = int(input("Enter the maximum number of cluster: "))
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

"""Plot the dendogram"""
import pylab
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (data['manufact'][id], data['model'][id], int(float(data['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')