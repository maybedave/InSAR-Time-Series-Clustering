#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions for PCA analysis

@author: Davide Festa
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

def std_scaler(array_to_scale):
    df_array_std = StandardScaler().fit_transform(array_to_scale)
    stdarray_mean = df_array_std[~np.isnan(df_array_std)].mean()
    stdarray_std = df_array_std[~np.isnan(df_array_std)].std()
    print("Mean value of scaled array is " + str(stdarray_mean))
    print("Std value of scaled array is " + str(stdarray_std))      
    return df_array_std, stdarray_mean, stdarray_std

def pca_algorithm(df_array_std):
    pca = decomposition.PCA()
    pca.fit(df_array_std)
    #checking the first 5 components
    plt.plot(pca.components_[0, :], linewidth=1, color='blue', label='PC1')
    plt.plot(pca.components_[1, :], linewidth=1, color='red', label='PC2')
    plt.plot(pca.components_[2, :], linewidth=1, color='green', label='PC3')
    plt.plot(pca.components_[3, :], linewidth=1, color='black', label='PC4')
    #plt.plot(pca.components_[4, :], linewidth=1, color='grey', label='PC5')
    plt.legend()
    plt.title('PCA eigenvectors')
    plt.xlabel('satellite acquisitions')
    plt.ylabel('LOS displacement [m]')
    plt.savefig('PCA_eigenvectors.png')
    
    tot_variance = pca.explained_variance_
    variance_perc = (pca.explained_variance_/sum(pca.explained_variance_))*100
    #print("PCA explained variance is:\n" + str(tot_variance))
    print("PCA explained ratio variance is:\n" + str(variance_perc))
    return tot_variance, variance_perc

def required_percentage():
    value = input("Please enter the required percentage (integer): ")
    return value

def components_accounting_for_required_perc_variance(variance_percentage, value):
    for i in range(0, len(variance_percentage)):
        sum(variance_percentage[0:i])
        if sum(variance_percentage[0:i]) >= int(value):
            break
    print("Components accounting for <=90% of variance : " + str(i))
    return