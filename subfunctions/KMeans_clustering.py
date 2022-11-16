#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means Clustering

@author: Davide Festa
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from collections import Counter


def Euclidean_KMeans(df_array, number_of_components):
    Kmeans = TimeSeriesKMeans(n_clusters=number_of_components, max_iter=5,
                      metric='euclidean', random_state=0).fit(df_array)
    cluster_center = Kmeans.cluster_centers_
    cluster_center_shape = Kmeans.cluster_centers_.shape
    time_series_class = Kmeans.predict(df_array)
    labels = Kmeans.labels_
    count_labels = list(Counter(labels).values())
    inertia = Kmeans.inertia_
    return Kmeans, cluster_center, cluster_center_shape, time_series_class, labels, count_labels, inertia

def cluster_distribution_plotter(number_of_components, count_labels, labels):
    labels_for_plot = list(range(1,number_of_components + 1))
    fig1, ax1 = plt.subplots()
    ax1.pie(count_labels, labels=labels_for_plot, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("% of points distribution per clusters")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.75, -1, 'no_of_samples='+str(len(labels)),
         verticalalignment='top', bbox=props)
    return plt.show()

def cluster_center_plotter(df_array, number_of_components, time_series_class, cluster_center):
    cluster_10th_percentile = []
    cluster_90th_percentile = []
    cluster_50th_percentile = []
    
    for yi in range(number_of_components):
        for time in range(df_array.shape[1]):
            class_temporary = df_array[time_series_class == yi]
            number_1 = np.percentile(class_temporary[:, time], 10)
            number_2 = np.percentile(class_temporary[:, time], 90)
            number_3 = np.percentile(class_temporary[:, time], 50)
            cluster_10th_percentile.append(number_1)
            cluster_90th_percentile.append(number_2)
            cluster_50th_percentile.append(number_3)
    cluster_10th_percentile = np.reshape(
        cluster_10th_percentile, (number_of_components, df_array.shape[1]))
    cluster_90th_percentile = np.reshape(
        cluster_90th_percentile, (number_of_components, df_array.shape[1]))
    cluster_50th_percentile = np.reshape(
        cluster_50th_percentile, (number_of_components, df_array.shape[1]))
   
    
    plt.figure()
    for yi in range(number_of_components):
        #plt.subplot(1, components,yi+1)
        # plt.subplot(round(components/3), round(components/3),yi+1) #plt.subplot(1, components,yi+1)
        plt.subplot(3, 3, yi+1)
        #fig, ax = plt.subplots()
        for xx in range(3):
            #ax.plot(x, y_est, '-')
            plt.plot(cluster_10th_percentile[yi, :], "k-", alpha=.2)
            plt.plot(cluster_90th_percentile[yi, :], "k-", alpha=.2)
            plt.plot(cluster_center[yi].ravel(), "r-")
            #plt.fill_between(time,km.cluster_centers_[yi].ravel()+cluster_10th_percentile[yi].ravel(),km.cluster_centers_[yi].ravel()-cluster_90th_percentile[yi].ravel(), color='grey', alpha=0.2)
            #plt.text(0.55, 0.85, 'Cluster %d' % (yi+1),
            #         transform=plt.gca().transAxes)
            
            if yi == 1:
                plt.title("Euclidean $k$-means")
    return cluster_50th_percentile

