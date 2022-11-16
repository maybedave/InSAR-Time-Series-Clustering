#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the reprojection of TS-InSAR LOS data into vertical and 
horizontal components by means of a regular spaced grid

@author: Davide Festa
"""
import numpy as np
import pandas as pd
import geopandas
import shapely
from pandas import Timedelta
import math as mt


# function creating grid where to interpolate points
# input:dataframe containing lats and lons of both asc and desc points, list of cellsizes
# output:polygonal geodataframe containing cells grid
def gridding(Lats_Lons_totalPoints, cell_size):
    gdf = geopandas.GeoDataFrame(Lats_Lons_totalPoints, 
            geometry=geopandas.points_from_xy(Lats_Lons_totalPoints.X, Lats_Lons_totalPoints.Y))
    gdf = gdf.drop(columns=['X', 'Y'])
    xmin, ymin, xmax, ymax= gdf.total_bounds
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0-cell_size
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1)  )
    cell = geopandas.GeoDataFrame(grid_cells, columns=['geometry'])
    
    return cell
    
# function dissolving and mediating Asc and Desc time series within every cell
# input: cells grid, dataframe of coordinates ascending, dataframe of coordinates descending
# output: grid with asc and desc data merged
def grid_dissolve(cell, dfcoords_asc, dfcoords_desc):
    
    gdf_dfcoords_asc=geopandas.GeoDataFrame(dfcoords_asc, geometry=geopandas.points_from_xy(dfcoords_asc.X, dfcoords_asc.Y))
    gdf_dfcoords_asc=gdf_dfcoords_asc.drop(columns=["X", "Y"])
    
    gdf_dfcoords_desc=geopandas.GeoDataFrame(dfcoords_desc, geometry=geopandas.points_from_xy(dfcoords_desc.X, dfcoords_desc.Y))
    gdf_dfcoords_desc=gdf_dfcoords_desc.drop(columns=["X", "Y"])
    
    merge_asc = geopandas.sjoin(gdf_dfcoords_asc, cell, how="left", op="within")
    dissolve_asc = merge_asc.dissolve(by="index_right", aggfunc="mean")
    cell_dissolve_asc = pd.merge(cell, dissolve_asc, left_index=True, right_index=True).drop(columns=["geometry_y"])
        
    merge_desc =geopandas.sjoin(gdf_dfcoords_desc, cell, how="left", op="within")
    dissolve_desc = merge_desc.dissolve(by="index_right", aggfunc="mean")
    cell_dissolve_desc = pd.merge(cell, dissolve_desc, left_index=True, right_index=True).drop(columns=["geometry_y"])
    
    datasets_names=["data_asc", "data_desc"]
    grid_with_asc_desc = dict(zip(datasets_names, [cell_dissolve_asc, cell_dissolve_desc]))
    
    return grid_with_asc_desc


# function resampling Asc and Desc time series by interpolation (choose the interpolation method at line 121)
# the number of time steps is automatically calculated as (total no. of days covered)/(no. of asc and desc acquisitions)
# input: asc formatted dates, desc formatted dates, grid with asc and desc data merged
# output: resampled grid, new interpolated dates
def resampling_ts(datelist_asc, datelist_desc, grid_with_asc_desc):
    t0asc=sorted(datelist_asc)[0] ; t0desc=sorted(datelist_desc)[0] ; tlast_asc = sorted(datelist_asc)[-1]; tlast_desc = sorted(datelist_desc)[-1]
    
    if t0asc > t0desc:
        
        start_date=t0asc
        
    else:
        
        start_date=t0desc #defining the upper date boundary (the upper overlapping date between ascendent and descendent dataset)
        
    if tlast_asc < tlast_desc:
        
        end_date=tlast_asc
        
    else:
        
        end_date=tlast_desc #defining the lower date boundary (the lowest overlapping date between ascendent and descendent dataset)
    
    keys_grid_asc=list(grid_with_asc_desc["data_asc"].index)
    keys_grid_asc=[int(x) for x in keys_grid_asc]
    grid_asc = grid_with_asc_desc["data_asc"] ; grid_asc = grid_asc.drop(columns=["geometry_x"])
    grid_asc=grid_asc.to_dict(orient="index")
    
    for e in range(len(keys_grid_asc)):
        grid_asc[keys_grid_asc[e]]=pd.DataFrame.from_dict(grid_asc[keys_grid_asc[e]], orient='index')
        grid_asc[keys_grid_asc[e]]=grid_asc[keys_grid_asc[e]].loc[start_date:end_date]
        #grid_asc[keys_grid_asc[e]].iloc[0]=0.0
    
    keys_grid_desc=list(grid_with_asc_desc["data_desc"].index)
    keys_grid_desc=[int(x) for x in keys_grid_desc]
    grid_desc = grid_with_asc_desc["data_desc"] ; grid_desc = grid_desc.drop(columns=["geometry_x"])
    grid_desc=grid_desc.to_dict(orient="index")
    
    for h in range(len(keys_grid_desc)):
        grid_desc[keys_grid_desc[h]]=pd.DataFrame.from_dict(grid_desc[keys_grid_desc[h]], orient='index')
        grid_desc[keys_grid_desc[h]]=grid_desc[keys_grid_desc[h]].loc[start_date:end_date]
    
    final_grid = {}
    for keys in keys_grid_asc:
        if keys in keys_grid_desc:
            final_grid[keys]=pd.merge(grid_asc[keys], grid_desc[keys], left_index=True, right_index=True, how="outer")
            final_grid[keys].iloc[0]=0.0
        
    for key in final_grid:
        final_grid[key].iloc[0]=0.0
        final_grid[key].columns=["Asc_TS", "Desc_TS"]
    
    first_key_of_final_grid=list(final_grid.keys())[0]
    oidx = final_grid[first_key_of_final_grid].index
    oidx=pd.to_datetime(oidx)
    frequency=Timedelta((oidx.max() - oidx.min())/ int(len(oidx))); frequency=Timedelta.round(frequency, "D")
    nidx = pd.date_range(oidx.min(), oidx.max(), freq=frequency)
    
    resampled_grid = {}
    for key in final_grid:
        resampled_grid[key] = final_grid[key].reindex(oidx.union(nidx)).interpolate(method="linear").reindex(nidx)
    
    return resampled_grid, nidx

# fuction for reprojection of Asc and Desc displacements along Vertical and Horizontal components
# input: resampled grid, Asc LOS angle (radians), Desc LOS angle (radians)

def reprojection(resampled_grid, asc_LOS_angle, desc_LOS_angle):
    asc_LOS_angle=asc_LOS_angle*(-1)
    cos_asc_LOS=mt.cos(asc_LOS_angle); sen_asc_LOS=mt.sin(asc_LOS_angle)
    #cos_asc_orbit=mt.cos(asc_orbit_angle); sen_asc_orbit=mt.sin(asc_orbit_angle)
    cos_desc_LOS=mt.cos(desc_LOS_angle); sen_desc_LOS=mt.sin(desc_LOS_angle)
    #cos_desc_orbit=mt.cos(desc_orbit_angle); sen_desc_orbit=mt.sin(desc_orbit_angle)
    
    for key,value in resampled_grid.items():
        resampled_grid[key]["Vv_TS"]=((resampled_grid[key]["Desc_TS"])-((resampled_grid[key]["Asc_TS"])*(sen_desc_LOS/sen_asc_LOS)))/((cos_desc_LOS)-((cos_asc_LOS*sen_desc_LOS)/sen_asc_LOS))
        resampled_grid[key]["Vh_TS"]=((resampled_grid[key]["Asc_TS"])-(resampled_grid[key]["Vv_TS"]*cos_asc_LOS))/sen_asc_LOS
    
    return print("reprojection successfull")
