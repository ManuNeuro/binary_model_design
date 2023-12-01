# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:15:43 2021

@author: Emmanuel Calvet
"""
from random import gauss
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import numpy as np
import pandas as pd
from pandas import Series

def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1, plot=False):
    b, a = butter_lowpass(cutoff, fs, order=order)
    if plot:
        # Plot the frequency response.
        plt.figure()
        w, h = freqz(b, a, worN=8000)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
    y = lfilter(b, a, data)
    return y


def whiteNoiseGauss(nbSample, filterLP=False, **kwargs):
    if filterLP:
        order = kwargs.get('order', 1) # Order of the filter
        fs = kwargs.get('SampleRate', 50) # sample rate, Hz
        cutoff = kwargs.get('cutoff', 1.0) # desired cutoff frequency of the filter, Hz
        plot = kwargs.get('plotFrequency', False)
        filt = lambda x: butter_lowpass_filter(x, cutoff, fs, order, plot)
    elif not filterLP:
        filt = lambda x: x
    return np.array(filt([gauss(0.0, 1.0) for i in range(nbSample)]))

def wn_scv(length=50, delay=6):
    time = np.arange(length-delay)
    wn_input = whiteNoiseGauss(length)
    wn_output = wn_input[:length-delay]
    wn_output = wn_output[:len(wn_output)-delay]
    wn_input = wn_input[:length-delay]
    nans = np.array([np.nan] * delay)
    wn_output = np.concatenate((nans, wn_output))
    
    plt.plot(time, wn_input, '--', color='b')
    plt.plot(time, wn_output, color='dodgerblue')

    dic_wn_versus_delay = {'time':time,
                           'wn_input':wn_input,
                           'wn_output':wn_output}
    df = pd.DataFrame(dic_wn_versus_delay)
    df = df.set_index('time')
    df.to_csv('white-noise.csv')

