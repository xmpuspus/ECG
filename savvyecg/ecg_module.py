import numpy as np
import pandas as pd
import os
import pickle
from .ecg_utils import *
import scipy.signal as ss
from sklearn.externals import joblib as jl
from biosppy.signals import ecg as ecgsig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Signal():
    def __init__(self, data, fs, hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=40):
        self.data = np.array(data)
        self.fs = fs
        self._hpass_cor_freq_hz = hpass_cor_freq_hz
        self._lpass_cor_freq_hz = lpass_cor_freq_hz
        
        # Preprocess the data
        self.df = make_ecg_dataframe(self.data, self.fs, col_name = 'ecg')
        self.df, self.rpeaks = process_ecg_df(self.df, fs, lp_cornerfreq = self._lpass_cor_freq_hz, 
                                              input_col = 'ecg', output_col = 'processed', 
                                              get_rpeaks = True, rpeak_col = 'r_peak')
        self.processed = self.df["processed"].values
        
    def get_heartrate(self, unit='permin'):
        self.heartrate = heart_rate(self.rpeaks, self.fs, unit=unit)
        return self
    
    def get_heartratevar(self, unit='ms'):
        self.heartratevar = heart_rate_var(self.rpeaks, self.fs, unit=unit)
        return self    
    
    def get_edr(self):            
        self.edr = edr(self.df.processed, self.rpeaks)
        return self

    def get_resprate(self, unit='permin'):
        self.resprate = resp_rate(self.edr[0], self.fs, unit=unit)
        return self
  
    # Divide whole signal into segments
    def add_segment_id(self, window, trunc_end = True):
        index_vals = self.df.index.values
        segments = divide_segments(index_vals, window)
        segment_id = np.hstack((np.ndarray.flatten(np.repeat(np.arange(segments.shape[0]).reshape(-1, 1), 
                                                             window, axis = 1)), 
                                np.ones(len(index_vals) - np.prod(segments.shape))*(segments.shape[0])))
        self.df['segment_id'] = np.array(segment_id, dtype = int)
        
        # Compile all segments in a single dataframe
        segments_df = pd.DataFrame(self.df.groupby('segment_id')['processed'].apply(list))
        
        if len(segments_df.iloc[-1]['processed']) < window:
            if trunc_end:
                print('Last segment is shorter than window size was removed.')
                self.segments_df = pd.DataFrame(segments_df.iloc[:-1])
            else:
                print('Last segment is shorter than window size was retained.')
                self.segments_df = pd.DataFrame(segments_df)
        else:
            self.segments_df = pd.DataFrame(segments_df)        
        return self
    
    def get_segment_features(self, beat_duration = 100e-3, rate_unit = 'permin',
                            get_beats = True):
        df_ = self.segments_df.copy()
        # R peaks
        seg_rpeaks = df_.processed.apply(lambda x: r_peak_loc(np.array(x), self.fs))
        self.segments_df['r_peaks'] = seg_rpeaks        
        
        # Heart rate and heart rate variability
        seg_hr = [heart_rate(x, self.fs, unit=rate_unit) for x in seg_rpeaks]
        seg_hrv = [heart_rate_var(x, self.fs) for x in seg_rpeaks]
        self.segments_df['hr_permin'] = seg_hr        
        self.segments_df['hrv_ms'] = seg_hrv
        
        # EDR and respiration rate
        seg_edr = [edr(np.array(df_.ix[i].processed), np.array(seg_rpeaks[i], dtype = int))[0] 
                   for i in df_.index.values]
        seg_rr = [resp_rate(x, self.fs, unit=rate_unit) for x in seg_edr]
        self.segments_df['edr'] = seg_edr
        self.segments_df['rr_permin'] = seg_rr
        
        # Beats
        if get_beats:
            seg_beats = [ecg_beats(np.array(df_.ix[i].processed), seg_rpeaks[i], 
                               self.fs, dt=beat_duration) for i in df_.index.values]
            self.segments_df['beats'] = seg_beats
            beats_rp = []
            beats_mag = []
            for b in seg_beats:
                if len(b)>0:
                    beatrp, beatmag = map(list, zip(*b))
                else:
                    beatrp = []
                    beatmag = []
                beats_rp.append(beatrp)
                beats_mag.append(beatmag)
            self.beats_df = pd.DataFrame({'segment_id': self.segments_df.index.values,
                                         'beat_rpeak': beats_rp, 'beat_mag': beats_mag}).set_index('segment_id')
        return self
    
    
    
###################################################
###################################################

#Models
vfib_model = jl.load('/Users/user/Documents/ecg_research/Classifiers/VFIB_Classifier.sav')
afib_model = jl.load('/Users/user/Documents/ecg_research/Classifiers/AFIB_Classifier.sav')
pac_pvc_bbbb_model = jl.load('/Users/user/Documents/ecg_research/Classifiers/PAC_PVC_BBBB_Classifier.sav')
apnea_model = jl.load('/Users/user/Documents/ecg_research/Classifiers/Apnea_Classifier.sav')

def predict_vfib_per_beat(beat):
    prediction = vfib_model.predict(beat)
    return prediction
    
    
def predict_vfib(ecg, f = 250):
    window = int(f * 0.4) #800ms beat
    ecg_new = baseline_correct(ecg, f)
    rpeaks = np.array(ecgsig.hamilton_segmenter(ecg_new, f)['rpeaks'])[2:-2] #detect rpeaks from ecg
    beats = np.array([ecg_new[i - window: i + window] for i in rpeaks])
    pca = PCA(n_components=10)
    beats_new = pca.fit_transform(beats)
    #beats_new = StandardScaler().fit_transform(beats_new)
    predictions = predict_vfib_per_beat(beats_new)
    dict_all = {}
    dict_all['beats'] = beats
    dict_all['PCA beats'] = beats_new
    dict_all['predictions'] = predictions
    
    return dict_all

def predict_afib_per_beat(beat):
    prediction = afib_model.predict(beat)
    return prediction


def predict_afib(ecg, f = 120):
    window = int(f * 0.4) #800ms beat
    ecg_new = baseline_correct(ecg, f)
    rpeaks = np.array(ecgsig.hamilton_segmenter(ecg_new, f)['rpeaks'])[2:-2] #detect rpeaks from ecg
    beats = np.array([ecg_new[i - window: i + window] for i in rpeaks])
    pca = PCA(n_components=20)
    beats_new = pca.fit_transform(beats)
    beats_new = StandardScaler().fit_transform(beats_new)
    predictions = predict_afib_per_beat(beats_new)
    dict_all = {}
    dict_all['beats'] = beats
    dict_all['PCA beats'] = beats_new
    dict_all['predictions'] = predictions
    
    return dict_all


def predict_pac_pvc_bbbb_per_beat(beat):
    prediction = pac_pvc_bbbb_model.predict(beat)
    return prediction


def predict_pac_pvc_bbbb(ecg, f = 250):
    window = int(f * 0.4) #800ms beat
    ecg_new = baseline_correct(ecg, f)
    rpeaks = np.array(ecgsig.hamilton_segmenter(ecg_new, f)['rpeaks'])[2:-2] #detect rpeaks from ecg
    beats = np.array([ecg_new[i - window: i + window] for i in rpeaks])
    pca = PCA(n_components=10)
    beats_new = pca.fit_transform(beats)
    beats_new = StandardScaler().fit_transform(beats_new)
    predictions = predict_pac_pvc_bbbb_per_beat(beats_new)
    dict_all = {}
    dict_all['beats'] = beats
    dict_all['PCA beats'] = beats_new
    dict_all['predictions'] = predictions
    return dict_all

def predict_apnea_per_segment(segment):
    prediction = apnea_model.predict(segment)
    return prediction


def predict_apnea(ecg, f = 100, segment_size = 60):
    
    window = int(f * segment_size) #800ms beat
    ecg_wind  = segment_ecg(ecg, segment_size = segment_size, f = f)
    rpeaks = np.array([ecgsig.hamilton_segmenter(i, f)['rpeaks'] for i in ecg_wind]) #detect rpeaks from ecg    
    RR_int = np.array([np.diff(i) for i in rpeaks])
    peak_count = np.array([len(i) for i in rpeaks])
    
    #remove segments without detectable rpeaks
    n_peaks = 30
    ecg_wind = ecg_wind[np.where(peak_count > n_peaks)]
    RR_int = RR_int[np.where(peak_count > n_peaks)]
    rpeaks = rpeaks[np.where(peak_count > n_peaks)]
    peak_count_new = peak_count[np.where(peak_count > n_peaks)]
    #features
    X = calculate_features(ecg_wind, rpeaks, peak_count_new, RR_int, f = f)
    X_c = StandardScaler().fit_transform(X)
    predictions = predict_apnea_per_segment(X_c)
    dict_all = {}
    dict_all['ecg'] = ecg_wind
    dict_all['predictions'] = predictions
    return dict_all