from __future__ import division
import os
import json
import numpy as np
import pandas as pd
import scipy.signal as ss
import scipy.interpolate as si
import seaborn as sns
from scipy.ndimage import convolve1d
import itertools
from imblearn.over_sampling import SMOTE
from biosppy.signals import ecg
import sensorlib.freq as slf
from biosppy.signals import ecg as ecgsig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


## process ECG signal 

def baseline_correct(x, fs):
    """
    Removes baseline wander of ECG signal
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    Returns
    -------
    corrected: array
        Array of containing the baseline-corrected signal
    
    """
    
    med_filt = ss.medfilt(ss.medfilt(x, kernel_size=int(200e-3*fs+1)), kernel_size = int(600e-3*fs+1))
    corrected = x-med_filt
    return corrected

    
def smooth(x, fs, order = 1, btype = 'low', corner_freq_hz = 150):
    """
    Smoothens signal using Butterworth filter
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    order: int, optional
        The order of the filter

    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    
    corner_freq_hz: scalar
        Number corresponding to the critical frequency of the filter
        
    Returns
    -------
    filtered: array
        Smoothened signal
        
    """
    nyquist = fs / 2.0
    f_c = np.array([corner_freq_hz, ], dtype=np.float64)  # Hz
    # Normalize by Nyquist
    f_c /= nyquist
    # Second order Butterworth low pass filter at corner frequency
    b, a = ss.butter(order, f_c, btype=btype)
    # Apply the filter forward and backward to eliminate delay.
    filtered = ss.filtfilt(b, a, x)
    return filtered


def preprocess_ecg(x, fs, lp_cornerfreq = 40):
    """
    Performs pre-processing of the raw ECG signal (High pass filter, 
    Low pass filter and Baseline correction)
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
                
    lp_cornerfreq: scalar, optional
        Number corresponding to the corner frequency of the filter
        
    Returns
    -------
    x3: array
        Processed signal
    """

    x1 = smooth(x, fs, btype = 'high', corner_freq_hz = 0.5)
    x2 = smooth(x1, fs, corner_freq_hz = lp_cornerfreq)
    x3 = baseline_correct(x2, fs)
    return x3


def r_peak_loc(x, fs):
    """
    Gives location of R peaks
    
    Parameters
    ----------
    x : array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    Returns
    -------
    rloc: array
        Array containing the R-peak locations in the array
        
    """
    rloc = ecg.hamilton_segmenter(x, fs)['rpeaks']
    return rloc


def ecg_beats(sig, rpeaks, fs, dt = 100e-3):
    """
    Gives an array of ECG beats
    
    Parameters
    ----------
    sig: array_like
        Array containing magnitudes of ECG signal

    rpeaks: array_like
        Array containing the index of the R peaks in the signal
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    dt: float, optional
        Number corresponding to the time duration to be taken from to left 
        and to the right of the R peak location, must be in seconds
    
    Returns
    -------
    beats: array
        Contains the ECG beats where the magnitudes of each beat is given by
        every row
        
    """
    width = int(dt * fs)
    total_time = len(sig)/fs
    beats = np.array([sig[int(r - width): int(r + width)] for r in rpeaks if ((r - width) >0 and (r + width)< len(sig))])
    return beats


def make_ecg_dataframe(x, fs, col_name = 'ecg'):
    """
    Converts an array of ECG signal to a data frame
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    col_name: string, optional
        String that will be used as a column name in the data frame
        
    Returns
    -------
    df: pandas dataframe
        Daframe containing the magnitudes of the ECG signal in one columns
        and the corresponding time of in seconds
   
    """
    df = pd.DataFrame({col_name: x, '_time_sec':np.arange(len(x))/fs})
    return df


def process_ecg_df(df, fs, lp_cornerfreq = 40, input_col = 'ecg', output_col = 'processed', 
                   get_rpeaks = False, rpeak_col = 'r_peak_loc'):
    """
    Adds the processed ECG signal and the R peaks (optional) to the dataframe 
    
    Parameters
    ----------
    df: pandas data frame
        data frame containing the magnitudes of the raw ECG signal in one column

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    lp_cornerfreq: scalar, optional
        Number corresponding to the corner frequency of the filter to be applied   
        
    input_col: string, optional
        String that corresponds to the column name of raw ECG signal
        
    output_col: string, optional
        String that will be used as a column name in the data frame for the 
        processed signal
        
    get_rpeaks: bool, optional
        If this is set to True, the R peaks in the signal will be labeled as 1
        and others as 0
        
    rpeak_col: string, optional
        String that will be used as a column name in the data frame for the 
        R peaks
    
    """
    df[output_col] = preprocess_ecg(df[input_col], fs, lp_cornerfreq)
    sig = df[output_col].values
    if get_rpeaks:
        df[rpeak_col] = 0
        rpeak_index = r_peak_loc(sig, fs)
        df.loc[rpeak_index, rpeak_col] = 1
        return df, rpeak_index
    else:
        return df
        
def edr(sig, rpeaks, interp = 'linear'):
    """
    Calculate the ECG-derived respiration (EDR)
    
    Parameters
    ----------
    sig: array_like
        Array containing magnitudes of ECG signal
        
    rpeaks: array_like
        Array containing indices of the R peaks in the ECG signal
    
    interp: string, optional
        Specifies the kind of interpolation as a string(‘linear’, ‘nearest’, ‘zero’, 
        ‘slinear’, ‘quadratic’, ‘cubic')
        
    Returns
    -------
    edr_sig: array
        Array containing the interpolated ECG-derived respiration signal
        
    rpeaks_interp: array
        Array containing the interpolated R-peak locations
    
    """
    f = si.interp1d(rpeaks, sig[rpeaks], kind=interp)
    rpeaks_interp = np.arange(rpeaks[0], rpeaks[-1]+1)
    edr_sig = f(rpeaks_interp)
    edr_sig -= np.mean(edr_sig)
    return edr_sig, rpeaks_interp

def resp_rate(edr_sig, fs, unit = 'persec'):
    """
    Calculate an estimate of the respiration rate from EDR
    
    Parameters
    ----------
    edr_sig: array_like
        Array containing magnitudes of EDR signal       
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    unit: string, optional 
        Specifies unit of respiration rate ('persec' refers to breaths per second,
        and 'permin' refers to breaths per minute)
        
    Returns
    -------
    rr: float
        Estimate of the respiration rate
        
    """
    
    multiplier = {'persec': 1, 'permin': 60}
    rr = slf.fund_freq_by_corr(edr_sig, samp_freq = fs)*multiplier[unit]
    return rr
                
    
def heart_rate(r_peaks, fs, unit = 'persec'):
    """
    Calculate heart rate from the location of R peaks
    
    Parameters
    ----------
    rpeaks: array_like
        Array containing indices of the R peaks in the ECG signal  
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    unit: string, optional 
        Specifies unit of heart rate ('persec' refers to breaths per second,
        and 'permin' refers to breaths per minute)
        
    Returns
    -------
    hr: float
        Estimate of the heart rate
        
    """ 
    multiplier = {'persec': 1, 'permin': 60}
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    hr = 1/np.nanmean(dt) *multiplier[unit]
    return hr

def heart_rate_var(r_peaks, fs, unit = 'ms'):
    """
    Calculate heart rate from the location of R peaks
    
    Parameters
    ----------
    rpeaks: array_like
        Array containing indices of the R peaks in the ECG signal  
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    unit: string, optional 
        Specifies unit of heart rate ('s' refers to second,
        and 'ms' refers to millisecond)
        
    Returns
    -------
    hrv: float
        Estimate of the heart rate variability
        
    """     
    multiplier = {'s': 1, 'ms': 1000}    
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    hrv = np.nanstd(dt)*multiplier[unit]
    return hrv
    
## dividing the whole signal into segments
def array_rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def divide_segments(a, window, overlap = 0):
    window = int(window)
    rolled = array_rolling_window(a, window)
    slide_by = int(window*(1-overlap))
    return rolled[0::slide_by]

def ecg_beats(sig, rpeaks, fs, dt = 100e-3):
    width = int(dt * fs)
    total_time = len(sig)/fs
    beats = [(r, sig[int(r - width): int(r + width)]) for r in rpeaks 
             if ((r - width) >0 and (r + width)< len(sig))]
    return beats



##############################################################
##############################################################
##############################################################


def EDR_all(ecgs, rpeaks_idxs, peak_count_nz):
    """
    Calculate EDR for segmented ecg signal with its respective rpeak locations.
    Difference with edr() is that this function interpolates with respect to the
    maximum number of rpeaks in the segmented ecg signal.
    
    Parameters
    ----------
    ecgs: array_like
        2D array containing segmented ECG signal
        
    rpeaks_idxs: array_like
        Array containing indices of the R peaks in the segmented ECG signal  
    
    peak_count_nz: array_like
        Array containing the number of R peak locations for each segment of the
        segmented ECG signal
        
    Returns
    -------
    EDR: array_like
        Array containing ECG-derived respiration per segment
        
    """  
      
    EDR_list = []
    for i in range(rpeaks_idxs.shape[0]):
        
        cs_rpeak = interp1d(rpeaks_idxs[i], ecgs[i][rpeaks_idxs[i]])
        min_x = np.min(rpeaks_idxs[i])
        max_x = np.max(rpeaks_idxs[i])
        n = np.max(peak_count_nz)
        t = np.linspace(min_x, max_x, n)
        edr_ = cs_rpeak(t)
        edr_ -= np.mean(edr_) #center the signal
        #edr_, _ = edr(ecgs[i], rpeaks_idxs[i])
        EDR_list.append(edr_)
    EDR = np.array(EDR_list)
    return EDR

def iqr(x):
    """
    Calculate inter-quartile range
    
    Parameters
    ----------
    x: array_like
        Array containing numbers for IQR
        
    Returns
    -------
    iqr: float
        Inter-quartile Range
        
    """ 
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr

def mad(x):
    """
    Calculate mean absolute deviation
    
    Parameters
    ----------
    x: array_like
        Array containing numbers for MAD
        
    Returns
    -------
    mad: float
        mean absolute deviation
        
    """ 
    
    mean_x = np.mean(x)
    mean_adj = np.abs(x - mean_x)
    mad = np.mean(mean_adj)
    return mad

def sample_entropy(a):
    """
    Calculate 2nd order sample entropy
    
    Parameters
    ----------
    a: array_like
        Array containing numbers to calculate sample entropy
        
    Returns
    -------
    sp: float
        sample entropy
        
    """ 
    sp = np.abs(a[2] - a).max(axis=0)
    return sp

def segment_ecg(ecg, segment_size = 60, f = 100):
    """
    Segment ECG signal
    
    Parameters
    ----------
    ecg: array_like
        1D Array containing amplitude of ECG signal
        
        
    segment_size: int, optional 
        Specifies segment length (15-sec, 30-sec, or 60-sec)
        
    f: float, optional
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
       
    Returns
    -------
    ecg_segmented: array_like
        Estimate of the heart rate variability
        
    """ 
    
    ecg[ecg < 0.01] = 0.01
    ecg_len = ecg.shape
    divs =int(ecg_len[0] /(segment_size * f))
    ecg_new = baseline_correct(ecg, f)
    windowed_ecg = np.array_split(ecg_new[:divs * segment_size * f], divs)  
    ecg_segmented = np.array([i for i in windowed_ecg if len(i) == len(windowed_ecg[0])])
    return ecg_segmented
    
def calculate_features(ecg_wind, rpeaks, peak_count_new, RR_int, f = 100):
    """
    Calculate features for detecting sleep apnea
    
    The features contain statistical measures, morphology and periodogram
    of the ECG signal.
       
    Parameters
    ----------
    ecgs_wind: array_like
        2D array containing segmented ECG signal
        
    rpeaks: array_like
        Array containing indices of the R peaks in the segmented ECG signal  
    
    peak_count_new: array_like
        Array containing the number of R peak locations for each segment of the
        segmented ECG signal
        
    RR_int: array_like
        Array containing RR intervals
        
    f: float, optional
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
           
    Returns
    -------
    X: array_like
        Features
        
    """ 
    n = 5e-2  #in seconds
    edr_sig = EDR_all(ecg_wind, rpeaks, peak_count_new)
    mean_RR = np.array(list(map(np.mean, RR_int)))
    std_RR = np.array(list(map(np.std, RR_int)))
    NN50_1 = np.array([np.sum(np.diff(i) < f*n) for i in RR_int])
    NN50_2 = np.array([np.sum(np.diff(i) > f*n) for i in RR_int])
    pNN50_1 = np.divide(NN50_1, peak_count_new, dtype = float)
    pNN50_2 = np.divide(NN50_2, peak_count_new, dtype = float)
    max_RR = np.array(list(map(np.max, RR_int)))
    min_RR = np.array(list(map(np.min, RR_int)))
    SDSD = np.array(list(map(np.std, np.array([np.diff(i) for i in RR_int]))))
    median_RR = np.array(list(map(np.median, RR_int)))
    RMSSD = np.array(list(map(lambda x: np.sqrt(np.mean(np.square(x))), np.array([np.diff(i) for i in RR_int]))))
    iqr_RR = np.array(list(map(iqr, RR_int)))
    mad_RR = np.array(list(map(mad, RR_int)))
    sp_RRint = np.array(list(map(sample_entropy, RR_int)))
    psd = np.array(list(map(lambda x: np.abs(np.fft.fft(x)), edr_sig)))
    mean_EDR = np.array(list(map(np.mean, edr_sig)))
    std_EDR = np.array(list(map(np.std, edr_sig)))
    sp_EDR = np.array(list(map(sample_entropy, edr_sig)))
    #PCA
    pca = PCA(n_components=20)
    EDR_new = pca.fit_transform(edr_sig)
    psd_new = pca.fit_transform(psd)
    X = np.column_stack([peak_count_new, mean_RR, std_RR, NN50_1, NN50_2, pNN50_1, pNN50_2, EDR_new, psd_new, \
                    mean_EDR, std_EDR, sp_EDR, sp_RRint, max_RR, min_RR, SDSD, median_RR, RMSSD, mad_RR, iqr_RR])
    
    return X
