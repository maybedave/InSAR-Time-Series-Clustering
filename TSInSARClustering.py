#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS-InSAR resampling and reprojection of Asc and Desc data into Vertical and Horizontal gridded data
Execute a cluster analysis of TS-InSAR data by using PCA and K-Means
Decompose clustered time series through Linear Regression and Fast Fourier Transform

@author: Davide Festa (davide.festa@unifi.it)
***Version 10/11/2022

"""
import sys
sys.path.append('W:/Ubuntu/BGS/1.cluster/backup_26_04_2022')
from kneed import KneeLocator
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
# import dependent functions
import V_H_reprojection
import KMeans_clustering
import PCAnalysis
import regression_DFT

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Read CSV of the ascending geometry time series Insar dataset containing the "X" "Y" field projected coordinates
ASC = pd.read_csv(r'W:/Ubuntu/BGS/1.cluster/ascending_data_sample.csv', sep=',') #file provided in 'InSAR-Time-Series-Clustering/sample_dataset/sample_dataset/ascending_InSAR_sample.csv'

# Read CSV of the descending geometry time series Insar dataset containing the "X" "Y" field projected coordinates 
DESC = pd.read_csv(r'W:/Ubuntu/BGS/1.cluster/descending_data_sample.csv', sep=',') #file provided in 'InSAR-Time-Series-Clustering/sample_dataset/sample_dataset/descending_InSAR_sample.csv'

datasets_names = ["data_asc", "data_desc"]
# unique dictionary containing both ascending and descending dataframes
Dataset_asc_desc_dict = dict(zip(datasets_names, [ASC, DESC]))

# Setting up dictionaries containing Insar-ts-related info, where each one containing both ascending and descending datasets attributes
data_coordinates = {}
ts = {}
date_ts = {}
date_formatted = {}
df_coords = {}
lats = {}
lons = {}
df_array = {}

# Requirements:
# The time series-related columns starts with "D" and no other columns should start with "D"

for key in Dataset_asc_desc_dict:
    data_coordinates[key] = pd.DataFrame(
        Dataset_asc_desc_dict[key], columns=['X', 'Y'])
    ts[key] = Dataset_asc_desc_dict[key].loc[:, Dataset_asc_desc_dict[key].columns.str.startswith(
        'D')]
    date_ts[key] = list(ts[key].columns.values)
    date_ts[key] = [e[1:] for e in date_ts[key]]
    date_formatted[key] = np.array([dt.datetime.strptime(
        d, '%Y%m%d').date() for d in date_ts[key]])  # line for formatting date names
    date_formatted[key] = list(date_formatted[key])
    ts[key].columns = date_formatted[key]
    # dict with final asc/desc arrays containing X Y coordinates and time series
    df_coords[key] = pd.concat([data_coordinates[key], ts[key]], axis=1)
    lats[key] = pd.DataFrame(df_coords[key], columns=['X'])
    lons[key] = pd.DataFrame(df_coords[key], columns=['Y'])
    # dict with final asc/desc arrays containing only time series displacement values
    df_array[key] = df_coords[key].iloc[:, 2:].to_numpy()

# Retrieving lats and lons from both ascending and descending datataset to set up the extension of the fishnet
AscDesc_lats = pd.concat([lats["data_asc"], lats["data_desc"]])
AscDesc_lons = pd.concat([lons["data_asc"], lons["data_desc"]])
Tot_points = pd.concat([AscDesc_lats, AscDesc_lons], axis=1)

# -------------------------- ASSESSMENT OF OPTIMAL GRID SIZE ------------------
# Evaluation of the best trade-off size by finding the smallest grid size able to capture ascending and descending data
# The size is evaluated by plotting the results of totcells/commoncells (totcells is the total number of cells produced at the n size grid and commoncells is the number of cell containing asc and desc data) against the n range of size values
# The knee point of the plotted curve is the best grid size 

# Cellsize input list
cellsize=list(np.arange(15, 216, 10))
totcells=[]
totcells_to_commoncells=[]

for e in cellsize:
    cell = V_H_reprojection.gridding(Tot_points,e)
    grid_with_asc_desc = V_H_reprojection.grid_dissolve(
        cell, df_coords["data_asc"], df_coords["data_desc"])
    common_indexes=grid_with_asc_desc["data_asc"].index.intersection(grid_with_asc_desc["data_desc"].index)
    totcells.append(int(cell.count()))
    totcells_to_commoncells.append(int(cell.count())/len(common_indexes))

for_grid_size_asses=pd.DataFrame(
    {'cellsize': cellsize,
     'totcells': totcells,
     'totcells_to_commoncells': totcells_to_commoncells
    })

gridsize=KneeLocator(for_grid_size_asses["cellsize"], for_grid_size_asses["totcells_to_commoncells"],curve="convex", direction="decreasing")
gridsize.plot_knee()
best_grid_size=gridsize.knee

# TO MANUALLY OVERRIDE THE AUTOMATED GRID SIZE SELECTION:
# best_grid_size=

#------------------------------------------------------------------------------
 
cell=V_H_reprojection.gridding(Tot_points, best_grid_size)

# # Checking gridding results by plotting
#cell.plot(facecolor="none", edgecolor='grey')
#plt.scatter(Tot_points["X"], Tot_points["Y"], s=1)
#plt.show()

grid_with_asc_desc = V_H_reprojection.grid_dissolve(
    cell, df_coords["data_asc"], df_coords["data_desc"])

resampled_grid, new_dates = V_H_reprojection.resampling_ts(
    date_formatted["data_asc"], date_formatted["data_desc"], grid_with_asc_desc)

# checking for Asc and Desc newly interpolated time series
resampled_grid[int(list(resampled_grid.keys())[0])].plot(style='.-')

# fuction for reprojection of Asc and Desc displacements along Vertical and Horizontal components
# input: resampled grid, Asc LOS angle (radians), Desc LOS angle (radians), WHICH NEED TO BE ADJUSTED TO THE ACQUISITION GEOMETRY OF THE ANALYSED A-DINSAR DATASET
V_H_reprojection.reprojection(resampled_grid, 0.6735, 0.7525)

# Extraction of Vertical and Horizontal displacement time series for PCA and clustering analysis
Vv_df_array = []
Vh_df_array = []

for key in resampled_grid:
    Vv_df_array.append(resampled_grid[key]
                       ["Vv_TS"].to_numpy().reshape((1, -1)))
    Vh_df_array.append(resampled_grid[key]
                       ["Vh_TS"].to_numpy().reshape((1, -1)))

Vv_df_array = np.concatenate(Vv_df_array, axis=0)
Vv_df_array[:, 0] = 0.0
Vh_df_array = np.concatenate(Vh_df_array, axis=0)
Vh_df_array[:, 0] = 0.0

# Taking out cells with empty data
index_cells_geometry = [x for x in list(
    cell.index) if x in list(resampled_grid.keys())]
geometry_cell = cell.iloc[index_cells_geometry]

# check the grid after deleting the empty cells
geometry_cell.plot(facecolor="none", edgecolor='grey')


# --------------------------------------PCA analysis---------------------------
# PCA to constrain the number of clusters to use with the unsupervised clustering,
# see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# The analysis is repeated for Horizontal time series and Vertical time series

# df_array_std should be close to 1, stdarray_mean should be close to 0, stdarray_std is the standardised arrray
df_array_std, stdarray_mean, stdarray_std = PCAnalysis.std_scaler(
    Vh_df_array)  
df_array_std2, stdarray_mean2, stdarray_std2 = PCAnalysis.std_scaler(
    Vv_df_array)

# PCA_tot_variance is the matrix variance value computed, perc_variance is the same expressed in percentage
PCA_tot_variance, perc_variance = PCAnalysis.pca_algorithm(df_array_std)
PCA_tot_variance2, perc_variance2 = PCAnalysis.pca_algorithm(df_array_std2)

# # function asking computing no. of components to retain based on the total percentage of variance that we want to retain
#wanted_percentage = PCAnalysis.required_percentage()
#wanted_percentage2 = PCAnalysis.required_percentage()

# number of components retrieved by means of cut-off value in terms of variance percentage
#number_of_components = PCAnalysis.components_accounting_for_required_perc_variance(
   # perc_variance, wanted_percentage)  # no. of components resulting
#number_of_components2 = PCAnalysis.components_accounting_for_required_perc_variance(
   # perc_variance2, wanted_percentage2)  # no. of components resulting

# automatic method to find the optimal number of components based on the best trade-off value
# evaluated on the curve of cumulative percentage of variance, by using the KneeLocator function
kvh = KneeLocator(np.asarray([(range(len(perc_variance)))]).squeeze(), np.asarray(
    [perc_variance]).squeeze(), curve="convex", direction="decreasing")
kvv = KneeLocator(np.asarray([(range(len(perc_variance2)))]).squeeze(), np.asarray(
    [perc_variance2]).squeeze(), curve="convex", direction="decreasing")
# kvh.plot_knee()
# kvv.plot_knee()

number_of_components = kvh.knee
number_of_components2 = kvv.knee

# --------------------------------------Cluster analysis-----------------------
# for theory, see https://scikit-learn.org/stable/modules/clustering.html
# for parameters setting, https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html

km, cluster_center, cluster_center_shape, time_series_class, labels, count_labels, inertia = KMeans_clustering.Euclidean_KMeans(
    Vh_df_array, number_of_components)
km2, cluster_center2, cluster_center_shape2, time_series_class2, labels2, count_labels2, inertia2 = KMeans_clustering.Euclidean_KMeans(
    Vv_df_array, number_of_components2)

# plot the % of the clusters
KMeans_clustering.cluster_distribution_plotter(
    number_of_components, count_labels, labels)
plt.savefig('W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/Clusters_%_distributionVh.png')
KMeans_clustering.cluster_distribution_plotter(
    number_of_components2, count_labels2, labels2)
plt.savefig('W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/Clusters_%_distributionVv.png')

# Plotting the clusters centre
# for plotting, https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py
# taking the 10th and 90th percentile to display along with the clusters centre

Vh_clusters_centroids = KMeans_clustering.cluster_center_plotter(
    Vh_df_array, number_of_components, time_series_class, cluster_center)
plt.savefig('W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/Clusters_centersVh.png')
Vv_clusters_centroids = KMeans_clustering.cluster_center_plotter(
    Vv_df_array, number_of_components2, time_series_class2, cluster_center2)
plt.savefig('W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/Clusters_centersVv.png')

labels = labels+1
labels2 = labels2+1
labels_dataframe = pd.DataFrame(labels, columns=['cluster'])
labels_dataframe2 = pd.DataFrame(labels2, columns=['cluster'])

df_coords_clusters = pd.concat(
    [geometry_cell.reset_index(drop=True), labels_dataframe.reset_index(drop=True), pd.DataFrame(index_cells_geometry, columns=["cell_index"])], axis=1)
df_coords_clusters2 = pd.concat(
    [geometry_cell.reset_index(drop=True), labels_dataframe2.reset_index(drop=True), pd.DataFrame(index_cells_geometry, columns=["cell_index"])], axis=1)

# export cluster location as Shapefile
df_coords_clusters.to_file(
    'W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/cluster_horizontal_components.shp')
df_coords_clusters2.to_file(
    'W:/Ubuntu/BGS/1.cluster/backup_26_04_2022/cluster_vertical_components.shp')

#----------------------------------------- TIME SERIES DECOMPOSE --------------
# for linear regression theory, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# for Fast fourier Transform theory, see https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html

# Here the decomposition is applied to one of the resulting cluster centroid series
# To decompose the cluster centroid of interest you have to manually assign it in the following line:
Vh_clusters_centroid_1=pd.DataFrame(Vh_clusters_centroids[0,:], index=new_dates)

# function computing Linear regression and DFT for a time series
# input: dates (DatetimeIndex), dataframe containing cluster centroid discrete values
# output:  slope of the best-fit line, intercept of the best-fit line, rmse of the regression, dataframe containing the first 6 peaks of the spectral power with conversion of frequency to relative period
slope, intercept, rmse, powerspectrum = regression_DFT.LinRegression_DFT(new_dates, Vh_clusters_centroid_1)
