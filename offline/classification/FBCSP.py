from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import mne
from mne.decoding import CSP

class FBCSP(BaseEstimator,TransformerMixin):
    def __init__(self,sfreq,bands,n_classes, n_components=4,reg=None,log = True,norm_trace = False, verbose = False):
        self.sfreq = sfreq
        self.bands = bands
        self.n_classes = n_classes
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.verbose = verbose
        self._csp = None

    def fit(self,X,y):
        self._csp = []
        for (l,h) in self.bands:
            Xf=mne.filter.filter_data(X,self.sfreq,l,h,verbose=self.verbose)
            csps_this_band = []
            for k in range(self.n_classes):
                y_bin = (y==k).astype(int)
                csp = CSP(n_components=self.n_components,reg=self.reg,log=self.log,norm_trace=self.norm_trace)
                csp.fit(Xf,y_bin)
                csps_this_band.append(csp)
            self._csp.append(csps_this_band)
        return self

    def transform(self,X):
        feats = []
        for (l,h), csps_this_band in zip(self.bands, self._csp):
            Xf=mne.filter.filter_data(X,self.sfreq,l,h,verbose=self.verbose)
            for csp in csps_this_band:
                feats.append(csp.transform(Xf))
        return np.concatenate(feats,axis=1)





