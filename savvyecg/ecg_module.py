import numpy as np
import pandas as pd
import os
import pickle
from .ecg_utils import *

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
        
    # Compute physiological parameters
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
        