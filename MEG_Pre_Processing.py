#### MEG ASSR PRE-PROCESSING PIPELINE 

# MNE Software:
[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119
[2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X


import mne
import numpy as np
import scipy
from scipy import signal, stats
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import copy
import random
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing.ica import ICA, run_ica
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.preprocessing import maxwell_filter
from mne.viz import plot_projs_topomap
from mne.io import RawArray
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from mne.chpi import _get_hpi_info, _calculate_chpi_positions
import meg_functions as mf


##### LOAD DATA AND PREPARE MONTAGE
data_path = #path containing data#
#patient_ids = [1,2,3,4,5,6,7,8,9,13,17,18,19,20,21,22,23,24,28,29,30,32,34,35,36] #current ids in our workspace 
run =  50 #change this number as per the subjectid 
patient = "s%02d" % run
PATIENT = "S%02d" % run
fname = #join path and naming elements here#
raw = mf.load_MEG(data_path,fname)

##### ORGANISE SAVING LOCATIONS
ica_path = #savepath for ica data#
icaname =#name of subj then rawica.fif
save_path=#savepath
saveall =#path and file name then %s_all-epo.fif' %(patient))
save40 = op.join(save_path, '%s_40epo.fif' %(patient))
save30 = op.join(save_path, '%s_30epo.fif' %(patient))
save20 = op.join(save_path, '%s_20epo.fif' %(patient))

##### PREPARE RAW TEST DATA
# this can be used for testing pipeline parameters on a specific dataset
#snr_format = mf.SNRcreate_test_data(raw)
##### GET CHPI 
qq = mf.get_chpi_info(raw)
##### ORGANISE EVENTS
events = mne.find_events(raw, stim_channel='STI101',shortest_event=1)
events = events[1:,:] # this deletes the first event as it is only 1 sample 
tmin = -1.5 # epoch baseline 
tmax = 1.5 # epoch end
baseline = (None, 0) # baseline correct from beginning to zero time
mne.viz.plot_events(events)
###### ASSR EVENTS
event_id = {'40hz':204,'30hz':194,'20hz':248} 
##### FILTER - MAX AND BANDPASS 
raw = mf.raw_filter(raw,1,1,qq,1,'in') 
###### FIND BAD CHANNELS
print(raw.info['bads'])
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, exclude='bads')
reject = dict(mag=100e-12,grad=4000e-13) # reject extreme epochs
########## ICA ##########
method = 'extended-infomax'
eogch = 'EOG001' #eog001 or 2
ica = mf.ICA_decompose(raw,method=method,decim=3,variance=0.99,npcas=None,maxpcas=None,reject=reject,picks=picks)
mne.preprocessing.ICA.save(ica,icaname) 
ica = mf.ICA_artefacts(ica,raw,eogch=None,eog=0)
ica.plot_sources(raw) # plot projections of time series - continuous
manual_bad_comps = # add indices
print(sorted(ica.exclude))
ica.exclude.extend(manual_bad_comps) # add in the manually selected components
print(sorted(ica.exclude))
ica.plot_overlay(raw)
ica.apply(raw)
##### Final Clean-up and Save ####
post_ica_epochs_all = mne.Epochs(raw,events,event_id,tmin,tmax,proj=True,picks=picks,
                   baseline=baseline,preload=True,
                   reject=reject,add_eeg_ref=False,reject_by_annotation=True)
post_ica_epochs_all.plot()
#manual_bad_epochs = [] #add indices
#post_ica_epochs_all.drop(manual_bad_epochs,reason='user define',verbose=True)
mne.Epochs.save(post_ica_epochs_all,saveall)#!/usr/bin/env python2
#### NEW EPOCHING ROUTINE ########
post_ica_epochs_40 = post_ica_epochs_all['40hz']
mne.Epochs.save(post_ica_epochs_40,save40)#!/usr/bin/env python2
post_ica_epochs_30 = post_ica_epochs_all['30hz']
mne.Epochs.save(post_ica_epochs_30,save30)#!/usr/bin/env python2
post_ica_epochs_20 = post_ica_epochs_all['20hz']
mne.Epochs.save(post_ica_epochs_20,save20)#!/usr/bin/env python2
