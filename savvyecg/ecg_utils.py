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
import sensorlib.freq as slf
from biosppy.signals import ecg as ecgsig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, signaltonoise, entropy
from sklearn.metrics import confusion_matrix

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
    rloc = ecgsig.hamilton_segmenter(x, fs)['rpeaks']
    return rloc


def ecg_beatsstack(sig, rpeaks, fs, dt = 100e-3):
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

def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(3.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[2.5,2.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([3.5,3.5],[-0.5,2.5], '-k', lw=2)
    
#     ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)
    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pred Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(0,3,
            'Sensitivity: %.2f'%(tp*1./(tp+fn)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(1,3,
            'Specificity: %.2f'%(tn*1./(fp+tn)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()
    
    print ('Sensitivity:', tp*1./(tp+fn))
    print ('Specificity:', tn*1./(fp+tn))


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


## signal quality features


def filter_flat_sigs(sig, annot = None, std_tol = 1e-12):
    sigs_filtered = sig[np.nanstd(sig, axis = 1) > std_tol]
    if annot is None:
        return sigs_filtered
    else:
        annot_filtered = annot[np.nanstd(sig, axis = 1) > std_tol]
        return sigs_filtered, annot_filtered
        

def match_beat_annot(sig, rpeaks, annot, fs, dt = 100e-3):
    width = int(dt * fs)
    total_time = len(sig)/fs
    annotations = np.array(['']*len(sig))
    annotations[rpeaks] = annot
    
    beat_annot = np.array([annotations[r] for r in rpeaks if ((r - width) >0 and (r + width)< len(sig))])
    return beat_annot

def rel_power(ecg_sig, fs, num_freqbounds = (5, 15), denom_freqbounds = (5, 40)):
    powerspec = ss.periodogram(ecg_sig, fs)
    numerator = np.trapz((powerspec[1])[(powerspec[0] >= num_freqbounds[0]) * (powerspec[0] <=num_freqbounds[1])], 
                         dx = powerspec[0][1] - powerspec[0][0])
    denominator = np.trapz((powerspec[1])[(powerspec[0] >= denom_freqbounds[0]) * (powerspec[0] <=denom_freqbounds[1])], 
                           dx = powerspec[0][1] - powerspec[0][0])
    return numerator/denominator

def power_spec(ecg_sig, fs, bins = 5, fmax = 5):
    ecg_sig -= np.nanmean(ecg_sig)
    a, b = ss.periodogram(ecg_sig, fs = fs)
    b[0] = 0
    b = b/np.max(b)
    b = b[a<fmax] 
    a = a[a<fmax]
    
    return np.sum(np.reshape(b, (bins, -1)), axis = 1) 

def sig_energy(ecg_sig):
    ecg_sig -= np.nanmean(ecg_sig)
    ecg_sig /= np.max(abs(ecg_sig))
    energy = np.sum(np.square(ecg_sig))
    return energy

def permutation_entropy(time_series, m, delay):
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe

def normal_hr(sig, fs=360):
    stdv = np.std(sig)
    samp_clip = np.clip(sig, -stdv*0.7, stdv*0.7)
    samp_smt = smooth(samp_clip, fs, order = 6, corner_freq_hz=2)
    
    #first derivative
    ss_fd = ss.savgol_filter(samp_smt, window_length=7, polyorder=5, deriv=1)

    #power spectral density
    psd_1, psd_2 = ss.periodogram(ss_fd, fs)
    fnhr = psd_1[np.argmax(psd_2)]
    return fnhr

def rtor_duration(r_peaks, fs, unit = 'ms'):
    multiplier = {'s': 1, 'ms': 1000}    
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    rtor = np.mean(dt)*multiplier[unit]
    return rtor


down_sampler = pd.read_pickle("../data/signal_quality/pca_200msbeat.p")

def pca_feature(beats, top_comp = 5):
    beat_length = beats.shape[1]
    if beat_length < down_sampler.n_components:
        f_out = interp1d(np.arange(beat_length), beats, axis=1)
        beats = f_out(np.linspace(0, beat_length-1, down_sampler.n_components))
    
    beats_pca = down_sampler.transform(beats)
    pca_comps = np.mean(np.sum(beats_pca[:, :top_comp], axis = 1)/np.sum(beats_pca, axis = 1))
    return pca_comps

def mean_beat_energy(beats):
    beats /= np.max(np.ndarray.flatten(beats))
    energy = np.square(beats).sum(axis = 1)
    mean_energy = np.mean(energy)
    return mean_energy

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def maxmin_beat(beats):
    beats /= np.max(np.ndarray.flatten(beats))
    maxmin = np.max(beats, axis=1) - np.min(beats, axis=1)
    return np.mean(maxmin)

def sum_beat_energy(beats):
    beats /= np.max(np.ndarray.flatten(beats))
    return np.sum(np.square(np.sum(beats, axis=0)))


def get_features(df_, fs):    
    
    df = pd.DataFrame.copy(df_)
    
    print('Computing features...'),
    # features from statistics of magnitude of ECG signal
    df.loc[:, 'f_stddev'] = df.processed.apply(lambda x: np.nanstd(x))
    df.loc[:, 'f_kurtosis'] = df.processed.apply(lambda x: kurtosis(x))
    df.loc[:, 'f_skewness'] = df.processed.apply(lambda x: skew(x))
    df.loc[:, 'f_rms'] = df.processed.apply(lambda x: rms(x))
    df.loc[:, 'f_energy'] = df.processed.apply(lambda x: sig_energy(x))

    # features from power spectrum of signal
    df.loc[:, 'f_relpower'] = df.processed.apply(lambda x: rel_power(x, fs))
    df.loc[:, 'f_relbasepower'] = df.processed.apply(lambda x: rel_power(x, fs, num_freqbounds=(1, 40), 
                                                                  denom_freqbounds = (0, 40)))
    fbins, fmax = 10, 10
    powspec_vals = np.vstack(df.processed.apply(lambda x: power_spec(x, fs, bins=fbins, fmax=fmax)).values)
    for i in range(fbins):
        df.loc[:, 'f_powspec'+str(i)] = list(powspec_vals[:, i])

    
    # features from physiological parameters
    df.loc[:, 'f_rpeakcount'] = df.r_peaks.map(len)
    df.loc[:, 'f_nhr'] =  df.processed.apply(lambda x: normal_hr(x, fs))
    df.loc[:, 'f_hrv'] =  df.r_peaks.apply(lambda x: heart_rate_var(x, fs))
    df.loc[:, 'f_rtor'] =  df.r_peaks.apply(lambda x: rtor_duration(x, fs))
    df.loc[:, 'f_rr'] = df.apply(lambda x: [x.processed, x.r_peaks], axis = 1).apply(lambda x: edr(x[0],x[1]))\
                          .apply(lambda x: resp_rate(x[0], fs))
        
    df.loc[:, 'f_sumbe'] = df.beats.apply(lambda x: sum_beat_energy(np.array(x)))
        
    df.loc[:, 'f_pca'] = 0
    df.loc[df.beats.map(len)>0, 'f_pca'] = df.beats[df.beats.map(len)>0].apply(lambda x:pca_feature(np.array(x)))
    
#    df.loc[:, 'f_mbe'] = 0
    df.loc[df.beats.map(len)>0, 'f_mbe'] = df.beats[df.beats.map(len)>0].apply(lambda x: mean_beat_energy(np.array(x)))
    
    df.loc[:, 'f_maxminbeat'] = 0
    df.loc[df.beats.map(len)>0, 'f_maxminbeat'] = df.beats[df.beats.map(len)>0].apply(lambda x: maxmin_beat(np.array(x)))
     
    print('Done!')

    return df

def smote_all_minority(train_set, train_label):
    sm = SMOTE(kind='regular')

    majority = max(set(train_label), key=list(train_label).count)
    major_index = np.where(train_label==majority)[0]
    minorities = list(set(train_label) - set([majority]))
    
    majority_data = train_set[major_index, :]
    majority_labels = train_label[major_index]
    
    oversampled_data = {majority: majority_data}
    oversampled_labels = {majority: majority_labels}
    
    for i in minorities:
        minor_index = np.where(train_label==i)[0]
        reduced_features = np.concatenate((train_set[major_index, :], train_set[minor_index, :]))
        reduced_labels = np.concatenate((train_label[major_index], train_label[minor_index]))  
        train_set_smote, train_label_smote = sm.fit_sample(reduced_features, reduced_labels)
        
        new_minor_index = np.where(train_label_smote==i)[0]

        oversampled_data[i] = train_set_smote[new_minor_index, :]
        oversampled_labels[i] = train_label_smote[new_minor_index]
        
    return oversampled_data, oversampled_labels


def conf_mat_params_binary(true_labels, prediction):
    num_pos = np.sum(true_labels)
    num_neg = len(true_labels) - num_pos
    
    confmat = confusion_matrix(true_labels, prediction)

    tn, fp, fn, tp = np.ndarray.flatten(confmat)
    sens = tp/num_pos
    spec = tn/num_neg
    fpr = 1-spec
    fnr = 1-sens
    acc = (tp+tn)/len(true_labels)
    
    out_dict = {'confmat': confmat,
                'accuracy': acc,
                'sensitivity': sens,
                'specificity': spec,
                'falsepositiverate': fpr,
                'falsenegativerate': fnr}
    return out_dict

