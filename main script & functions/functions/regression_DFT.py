#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear regression + Discrete Fourier Transform

@author: Davide Festa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# function computing Linear regression and DFT for a time series
# input: dates (DatetimeIndex), dataframe containing cluster centroid discrete values
# output:  slope of the best-fit line, intercept of the best-fit line, rmse of the regression, dataframe containing the first 6 peaks of the spectral power with conversion of frequency to relative period
def LinRegression_DFT(dates, ts_array):
    X=np.array(range(len(dates)), dtype=int).reshape(-1, 1)
   # X_plot=pd.to_datetime(dates)
    
    slopes=np.empty([1])
    intercepts=np.empty([1])
    
    ##linear regression
    reg=LinearRegression().fit(X, ts_array) 
    predict_reg=reg.predict(X)
    slope=reg.coef_; np.append(slopes, slope)
    intercept=reg.intercept_; np.append(intercepts, intercept)
    mse = mean_squared_error(ts_array, predict_reg)
    rmse = np.sqrt(mse)
  
    ##Fast Fourier Trasform
    nobs = len(ts_array[0].tolist())
    ft = np.abs(np.fft.rfft(ts_array[0].tolist()))
    freq = np.fft.rfftfreq(nobs)
    
    ##plotting the results
    fig=plt.figure()
    ax1=plt.subplot(311)
    plt.plot(ts_array.index, ts_array.values)
    plt.ylabel('Displ[mm]')
    
    ax2=plt.subplot(312, sharey = ax1)
    text_toshow="Y={}x+{}".format(slope,intercept)
    ax2.text(0.5, 0.8, text_toshow, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    plt.plot(ts_array.index, predict_reg)
    plt.ylabel('Displ[mm]')
    
    ax3=plt.subplot(313)
    plt.plot(freq[2:], ft[2: ])
    # markerline, stemline, baseline, = ax3.stem(ft[2:],freq[2:],linefmt='k-',markerfmt='ko',basefmt='k.')
    # plt.setp(stemline, linewidth = 0.2)
    # plt.setp(markerline, markersize = 1)
    plt.xlabel('Frequency (1/day)')
    plt.ylabel('Amplitude')
    
    #printing power spectrum peaks
    ts_season = pd.DataFrame({'Amplitude':ft[2:], 'Frequency':freq[2:]})
    ts_season['Period in days'] = (1/ts_season['Frequency'])
    df=ts_season.sort_values(by=['Amplitude'], ascending=False).head(6)
    df=df.reset_index()
    df=df.rename(columns={'index':'F_label'})
    for i in range(0,len(df.index)):
        df.iloc[i:, 0]='F_'+str(i+1)
    
    return slope, intercept, rmse, df