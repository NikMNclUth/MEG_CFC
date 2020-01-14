"""
ASSR SOURCE Localisation 

--- This script demonstrates how to loop through the cleaned MEG data and conduct source localisation using
--- a defined region of interest.
--- Requires FreeSurfer Parcellated MRI
--- See MNE website for details on the creation of files for source localisation
--- path variables should be created appropriately for your own data
--- creates python and matlab outputs

Notes:
- improved snr (smaller regularisation parameter)

MNE CODE: 
[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119
[2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X

"""

###############################################################################
# Setup environment
import numpy as np
import matplotlib.pyplot as plt
import mne
import os.path as op
from mne.minimum_norm import (read_inverse_operator, make_inverse_operator, apply_inverse_epochs,
                              write_inverse_operator,apply_inverse)
from mne3.event import make_fixed_length_events
from scipy.fftpack import fftfreq
#from mne.datasets import sample
from mne3.time_frequency import csd_fourier,csd_morlet,csd_multitaper,tfr_morlet, psd_multitaper
from mne3.beamformer import tf_dics, make_dics,apply_dics_csd
from mne.viz import plot_source_spectrogram
import copy
import scipy.io
###############################################################################
######################### Filepaths ###########################################
###############################################################################
print(__doc__)
data_path = #path to cleaned data here
newdatapath = #path to save the data in
newimagepath = #path to save images in
###############################################################################
######################### Label Variables #####################################
###############################################################################
labeln_rh1 = 'transversetemporal-rh'
labeln_lh1 = 'transversetemporal-lh'
labeln_rh2 = 'superiortemporal-rh'
labeln_lh2 = 'superiortemporal-lh'


landmarks = ['transverse','superior']
###############################################################################
######################### DSPM Variables ######################################
###############################################################################
method = "dSPM"
snr = 3. # higher value lowers the regularisation parameter making the solution more sensitive to high amplitudes
lambda2 = 1. / snr ** 2
###############################################################################
######################### Processing Loop 40hz  ###############################
###############################################################################
excludeH = [] # subjects to exclude from the loop go in here
for run in range(1, 25): # use length of your data set to define the range
    if run in excludeH:
        continue
    #initialize names   
    subject = "h%02d" % run
    fssub = "H%02d" % run #freesurfer subject 
    subjects_dir=op.join('/path/path/path/path/', '%s/RAW/fs_test/' % (fssub)) #create subject directory for freesurfer data
    print("processing subject: %s" % subject)
    src_fname=op.join('/path/path/path/path/%s/RAW/fs_test/%s/bem/%s-oct-6-src.fif' %(fssub, fssub, fssub))
    inv40_fname=op.join(data_path,'inv/', '%s_40hz-meg-oct-6-inv.fif' % (subject))
    cov40_fname=op.join(data_path, 'noise_cov/', '%s_noise_40-cov.fif' % (subject))
    fname40=op.join(data_path, 'epochs/', '%s_40epo.fif' % (subject))
    fwd_fname40=op.join(data_path, 'fwd/', '%s_40.fwd.fif' % (subject))
    epochs40=mne.read_epochs(fname40, proj=True, preload=True, verbose=None)
    evoked40 = epochs40.average()
    fwd40 =  mne.read_forward_solution(fwd_fname40, surf_ori=True)
    noise_cov40=mne.read_cov(cov40_fname)
    inverse_operator40 = read_inverse_operator(inv40_fname)
    # label information
    # read label information
    label_rh = mne.read_labels_from_annot(fssub, parc='aparc',
                                   subjects_dir=subjects_dir,
                                   regexp=labeln_rh1)[0]

    label_lh = mne.read_labels_from_annot(fssub, parc='aparc',
                                   subjects_dir=subjects_dir,
                                   regexp=labeln_lh1)[0]
    
    label_rh2 = mne.read_labels_from_annot(fssub, parc='aparc',
                                   subjects_dir=subjects_dir,
                                   regexp=labeln_rh2)[0]
    
    label_lh2 = mne.read_labels_from_annot(fssub, parc='aparc',
                                   subjects_dir=subjects_dir,
                                   regexp=labeln_lh2)[0]
    # create label using superior and transverse temporal cortex10
    labelbothLH=label_lh+label_lh2
    labelbothRH=label_rh+label_rh2
    # run dspm using improved reg and wider roi
    stc40_anat_bothL = apply_inverse_epochs(epochs40, inverse_operator40, lambda2,label=labelbothLH,
                                           method=method, pick_ori=None)
    stc40_anat_bothR = apply_inverse_epochs(epochs40, inverse_operator40, lambda2,label=labelbothRH,
                                           method=method, pick_ori=None)
    
    stcs40_anat_bothL=list()
    for j in stc40_anat_bothL:
        data=j.data
        stcs40_anat_bothL.append(data)
        
    stcs40_anat_bothR=list()
    for j in stc40_anat_bothR:
        data=j.data
        stcs40_anat_bothR.append(data)
    del j, data
    
    scipy.io.savemat(newdatapath+fssub+'DSPMBoth40L.mat', {"stcs40_anat_bothL":stcs40_anat_bothL})
    scipy.io.savemat(newdatapath+fssub+'DSPMBoth40R.mat', {"stcs40_anat_bothR":stcs40_anat_bothR})
    np.save(newdatapath+fssub+'DSPMBoth40LHSTC', stc40_anat_bothL)
    np.save(newdatapath+fssub+'DSPMBoth40RHSTC', stc40_anat_bothR)
