# MEG Pre-Processing Module

# This module contains functions for the loading and pre-processing of MEG data. 
# An optional set of functions is also contained for evaluating different aspects of the signal to noise ratio.
# We anticipate that the SNR functions will be useful for assessing the parameters for specific datasets.

#MNE Code:
#[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119
#[2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X

# ENVIRONMENT
print('Defining environment')
import mne
import numpy as np
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
import bad_ic as bic

def load_MEG(data_path,fname):
    print('finding files and creating montage')
    raw_fname = op.join(data_path,fname)
    raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=False,preload=True)
    raw.montage=mne.channels.read_dig_montage(hsp=None,hpi=None,elp=None,point_names=None,
                                    unit='auto',fif=raw_fname,transform=True,
                                    dev_head_t=False)
    return raw 

def get_chpi_info(raw):
    raw.fix_mag_coil_types() # adjust magnetometer coil types
    print('Acquiring chpi information')
    quat=mne.chpi._calculate_chpi_positions(raw)
    quatnorm=quat[...,0]+(raw.first_samp/1000)
    qq = quat
    qq[:,0]=quatnorm 
    return qq

def raw_filter(raw,maxwell,bandpass,qq,chpi,reg):
    # note, reg is a string ('in') or None
    ctfile = '/raid5/rcho/PSYCH_CFC/MEG_analysis/ct_sparse.fif'
    ssscal = '/raid5/rcho/PSYCH_CFC/MEG_analysis/sss_cal.dat'
    import time
    if maxwell == 1 and len(qq)>0: #mf option 1 and qqfile provided
        print('performing Maxwell filtering with 10second TSSS window and head motion correction')
        time.sleep(2.0)
        mne.preprocessing.maxwell_filter(raw, origin='auto',coord_frame='head',
                                         cross_talk=ctfile,calibration=ssscal,
                                         regularize=reg,head_pos=qq,
                                         st_fixed = True, st_duration = 10.0)
    if maxwell == 2 and len(qq)>0: #mf option 2 and qqfile provided
        print('performing Maxwell filtering with 20second TSSS window and head motion correction')
        time.sleep(2.0)
        mne.preprocessing.maxwell_filter(raw, origin='auto',coord_frame='head',
                                         cross_talk=ctfile,calibration=ssscal,
                                         regularize=reg,head_pos=qq,
                                         st_fixed = True, st_duration = 20.0)
    if maxwell == 3 and len(qq)>0: #mf option 3 and qqfile provided
        print('performing Maxwell filtering with 30second TSSS window and head motion correction')
        time.sleep(2.0)
        mne.preprocessing.maxwell_filter(raw, origin='auto',coord_frame='head',
                                         cross_talk=ctfile,calibration=ssscal,
                                         regularize=reg,head_pos=qq,
                                         st_fixed = True, st_duration = 30.0)   
    if maxwell == 4 and len(qq)==0: #mf option 4 and no qqfile provided
        print('performing basic Maxwell filtering with no qq file')
        time.sleep(2.0)
        mne.preprocessing.maxwell_filter(raw, origin='auto',coord_frame='head',
                                         cross_talk=ctfile,calibration=ssscal,
                                         regularize=reg,head_pos=None,
                                         st_fixed = True)
    if maxwell == 4 and len(qq)>0: #mf option 5 and qqfile provided
        print('performing basic Maxwell filtering with  qq file - sss not tsss')
        time.sleep(2.0)
        mne.preprocessing.maxwell_filter(raw, origin='auto',coord_frame='head',
                                         cross_talk=ctfile,calibration=ssscal,
                                         regularize=reg,head_pos=qq,
                                         st_duration = None)    

    if chpi == 1: 
        #remove chpi and line noise (alternative to notch filtering)
        mne.chpi.filter_chpi(raw,include_line=True,verbose=None)
    if bandpass == 1:
        raw.filter(.01, 400.0, method='iir')
    return raw


import scipy.signal
def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

#def raw_events(raw,tmin,tmax):
#    events = mne.find_events(raw, stim_channel='STI101',shortest_event=1)
#    events = events[1:,:] # this deletes the first event as it is only 1 sample long
#    baseline = (None,0)
#    return raw

#def ICA_cleaning(raw,method,decim,variance,npcas,maxpcas,reject,eog,ecg,eogch,manual,picks,savepath,savename):
##    # inputs
##    # raw = raw data object
##    # method = ica method
##    # decim = decimation factor
##    # variance = %variance to reduce the ica estimation to (float)
##    # npcas = number of principal components (integer or boolean)
##    # maxpcas = maximum number of pcs (integer or boolean)
##    # comp_noiscov = if noiscov not provided determine whether or not to compute (1 = yes, 2 = no) --- future versions
##    # noiscov = noiscovariance matrix (variable or boolean) --- future versions
##    # reject = threshold to reject extreme epochs (float,dict, boolean)
##    # eog = detect eog components (1 = yes 2 = no)
##    # ecg = detect ecg components (1 = yes 2 = no)
##    # eogch = name of eogchannel
##    # manual = allow for manual inspection
##    IN FUTURE VERSIONS ADD AUTOMATIC DETECTION OF BEST %VARIANCE TO USE
##################### RUN ICA     
#     ica = ICA(n_components=variance, n_pca_components=npcas, max_pca_components=maxpcas,
#               method=method,verbose=True)
#     ica.fit(raw,decim=decim,reject=reject,picks=picks)
#     ica.get_sources(raw) 
#     icasave = savepath+savename
#     mne.preprocessing.ICA.save(ica,icasave)
##################### Artefact Detection    
#     if eog == 1:
#         eog_inds,scores = ica.find_bads_eog(raw,l_freq=1,h_freq=10)
#     if len(eog_inds)>0:
#         bad_ica_eogs = eog_inds
#     else: 
#         print('no eog components found, please use manual inspection')
#     if ecg == 1:
#         ecg_inds,scores = ica.find_bads_ecg(raw)
#     if len(ecg_inds)>0:
#         bad_ica_ecgs = ecg_inds
#     else: 
#         print('no ecg components found, please use manual inspection')
#     ica.detect_artifacts(raw,eog_ch=eogch,skew_criterion=range(2),
#                     kurt_criterion=range(2),var_criterion=range(2))
#     return ica
#     if manual == 1:
#         ica.plot_sources(raw)
#         manual_bad = input('Please Identify Bad Components: ')
#         ica.exclude.extend(manual_bad) 
#
#     return raw
def ICA_decompose(raw,method,decim,variance,npcas,maxpcas,reject,picks):
     r = np.random.RandomState(1234) # allow for reproducible results
     r.uniform(0, 10, 5)
    #################### RUN ICA     
     ica = ICA(n_components=variance, n_pca_components=npcas, max_pca_components=maxpcas,
               method=method,verbose=True,random_state = r)
     ica.fit(raw,decim=decim,reject=reject,picks=picks)
     ica.get_sources(raw) 
     return ica

#def ICA_artefacts(ica,raw,eogch,eog):
#    if eog == 1:
#        eog_inds,scores = ica.find_bads_eog(raw,l_freq=1,h_freq=10,ch_name=eogch)
#        if len(eog_inds)>0:
#            bad_ica_eogs = eog_inds
#        else: 
#            print('no eog components found, please use manual inspection')
#    else:
#        print('no eog channel please use manual inspection')
#
#    ecg_inds,scores = ica.find_bads_ecg(raw)
#    if len(ecg_inds)>0:
#        bad_ica_ecgs = ecg_inds
#    else: 
#        print('no ecg components found, please use manual inspection')
#    ica.detect_artifacts(raw,eog_ch=eogch,skew_criterion=range(2),
#                    kurt_criterion=range(2),var_criterion=range(2))
#    return ica
def ica_artefacts(ica,raw,eogch):
    ica.find_bads_eog(raw,l_freq=1,h_freq=10,ch_name=eogch)
    ica.find_bads_ecg(raw)
    ica.detect_artifacts(raw,eog_ch=eogch,skew_criterion=range(2),
                    kurt_criterion=range(2),var_criterion=range(2))
    borderact = bic.border_activation(ica,raw)
    borderact = np.argwhere(borderact>0)
#    borderact = np.squeeze(borderact)
    for iii in range(np.shape(borderact)[0]):
        ica.exclude.extend(borderact[iii])
    exc = np.unique(ica.exclude)
    exc = exc[:,np.newaxis]
    ica.exclude = []
    for iii in range(np.shape(exc)[0]):
        ica.exclude.extend(exc[iii])
    return ica



######################################################################################################
# SNR SPECIFIC FUNCTIONS
#1) snrcreate_test_data - generate indices for snr testing
        # returns:
            # random channel from list (name and integer)
            # 20 second array of data from random channel
            # 4 ASSR channels (name and integers)
def SNRcreate_test_data(raw):
    # stage 1 - generate random channel - this will be passed to the output and used in later stages
    import copy
    import random
    rchan = random.randint(0,raw.info['nchan'])
    rn = raw.info['ch_names']
    rchanName = rn[rchan]
    rtest = copy.deepcopy(raw)
    rtest2= rtest._data[rchan,20000:220000]
    del rtest
    # stag 2 - generate ASSR channel info - this will be used for signal to noise testing later on.
    ASSRchan1 = rn[176]
    ASSRnum1 = 176
    ASSRchan2 = rn[220]
    ASSRnum2 = 220
    ASSRchan3 = rn[50]
    ASSRnum3 = 50
    ASSRchan4 = rn[200]
    ASSRnum4 = 200
    ASSRroi = [176,220,50,200]
    return{'rand_chan':rchan, 'rand_chan_data':rtest2, 'rchanName':rchanName,
           'ASSRchan1':ASSRchan1,'ASSRchan2':ASSRchan2,'ASSRchan3':ASSRchan3,
           'ASSRchan4':ASSRchan4,'ASSRnum1':ASSRnum1,'ASSRnum2':ASSRnum2,
           'ASSRnum3':ASSRnum3,'ASSRnum4':ASSRnum4,'ASSRroi':ASSRroi} # create dictionary of outputs


def SNRwavelets(epochs_condition,low,high,step,timewindow,snr_format,numrois,frqwindow,snr_format_name):
#########################################################################################################################
# Based on SNR estimation in evoked responses described in Gonzale-Morino et al, (2014)
# SNRwavelets performs single trial and evoked response wavelet transformation on the specified, epoched data,
# and uses this information to provide an estimate of the SNR in the frequency range of interest, as well as 
# more broadly across all bands for induced and evoked power.

# Inputs:
    # epochs_condition = epoched data (object)
    # low = lowest frequency to estimate
    # high = highest frequency to estimate
    # step = interval between frequencies
    # timewindow = samples of interest for evoked response
    # snr_format = dictionary of roi channel information (see SNRcreate_test_data)
    # numrois = number of roi channels
    # frqwindow = frequencies defining the evoked response of interest (for ASSR this would be between 38 and 42 for example)
    # snr_format_name = string with names to find in the snr_format dict
# Returns:
    # dictionary of SNR values
    # roi_snr_ASSR - snr for each channel of the evoked response
    # roi_snr_EVOKEDbands - snr for each channel of the evoked bands
    # roi_snr_INDUCEDbands - snr for each channel of the induced bands

####### DEFINE ENVIRONMENT
    print('Importing additional modules')
    import scipy
    from scipy import stats
    import numpy as np
    import copy
    import mne
    from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
# Organise input and perform wavelet transform for single trials and average response data
    print('Beginning wavelet transforms')
    # frequency information for wavelets
    freqs = np.arange(low,high,step)
    n_cycles = freqs/4.
    # plot data - whole head - single trials
    power = mne.time_frequency.tfr_morlet(epochs_condition,freqs=freqs,
                                          n_cycles=n_cycles,use_fft=False,
                                          return_itc=False,decim=3,n_jobs=1,average=False)
    power.apply_baseline(mode='ratio', baseline=(-.5, 0)) # apply baseline correction using ratio method (power is divided by the mean baseline power)
     # plot data - whole head - average
    powerAV = mne.time_frequency.tfr_morlet(epochs_condition,freqs=freqs,
                                          n_cycles=n_cycles,use_fft=False,
                                          return_itc=False,decim=3,n_jobs=1,average=True)
    powerAV.apply_baseline(mode='ratio', baseline=(-.5, 0)) # apply baseline correction using ratio method (power is divided by the mean baseline power)
    # organise rois
    print('Extracting information from region of interest sites')
    rois = np.zeros(numrois,dtype = np.int)
    for x in range (0,np.shape(rois)[0]): # what this loop is doing is to go through and get the name of the items to select from the snr_format dict. 
    # this information is then used to find which items to use for the rois
        text1 = snr_format_name
        text2 = str(x+1)
        text3 = text1+text2
        rois[x]=int(snr_format[text3])
    eppower = copy.deepcopy(power.data[:,rois,:,]) # trials,channels,freqs,time
    eppowerAV = copy.deepcopy(powerAV.data[rois,:,]) # channels,freqs,time
    del power 
    del powerAV
    # we've now got the roi wavelet data. the next steps are to apply the appropriate baseline
    # and then to estimate the total power from the average of 39:41hz in the evoked and single trials
    # following this we can estimate the snr for each channel, and globally over our roi
##########################################################################################################################
# EVOKED RESPONSE SNR     
    # create evoked power average total (39:41hz)
#    windAV = eppowerAV[...,[18:20],[starta:enda]]
    print('Estimating SNR for evoked response')
    chAVpower = np.zeros(np.shape(rois)[0])
    for x in range(0,np.shape(rois)[0]):
        temp = eppowerAV[x,frqwindow,:]
        f=np.zeros(len(frqwindow))
        for y in range(0,len(frqwindow)):
            f[y] = sum(temp[y,timewindow[0]:(timewindow[-1]+1)])
        f = np.mean(f,0)
        del temp
        chAVpower[x]=f
    # create single trial power average total (39:41hz)
    chSTpower = np.zeros((np.shape(rois)[0],np.shape(eppower)[0]))
    for x in range(0,np.shape(eppower)[0]):
        temp1 = eppower[x,:,:,:]
        for y in range(0,np.shape(rois)[0]):
            temp2 = temp1[y,frqwindow,:]
            f=np.zeros(len(frqwindow))
            for yy in range(0,len(frqwindow)):
                f[yy] = sum(temp2[yy,timewindow[0]:(timewindow[-1]+1)])
            f = np.mean(f,0)
            chSTpower[y,x]=f
            del temp2
        del temp1
    # this is what we've all been waiting for, get channel snr
    chSNR_ASSR = np.zeros(np.shape(rois)[0])
    for x in range(0,np.shape(rois)[0]):
        temp1 = chAVpower[x]
        temp2 = chSTpower[x,:]
        snr = temp1/stats.sem(temp2)
        chSNR_ASSR[x]=snr
        del temp1
        del temp2
        del snr
# what this section is doing:
# the first loop goes through each of the roi channels and sums all of the power values in a given frq range over the specified time bin.
# this is then averaged across the rois.   
# the second set of loops goes through the single trials and performs the same procedure.
# the third loop estimates the SNR for each roi channel.     
#########################################################################################################
# Individual band SNR
# this section will take all of the individual frequency bands and estimate the SNR
# for evoked and induced power. Induced power is retained in the avergae response 
# by squaring individual power values (see Gonzale-Morino et al, 2014).
# In this version the window can be set to include any given time window but should
# be focussed on the task response period to allow for analysis of time and phase
# locked properties of the stimulus.

# Frequency bands definition
    delta = 0
    theta = np.arange(1,3,1)
    alpha = np.arange(3,7,1)
    beta = np.arange(7,20,1)
    gamma = np.arange(20,29,1)
# Evoked power
# pt1 - average response
    print('Estimating SNR for evoked response per band')
    ch = 0
    tempout = np.zeros((len(freqs),numrois))
    while ch<numrois:
        data = eppowerAV[ch,:,:]
        for x in range(0,len(freqs)):
            temp = data[x,timewindow[0]:(timewindow[-1]+1)]
            temp = np.sum(temp)
            tempout[x,ch]=temp
            del temp
        ch=ch+1
        del data
    evoked_bands_pt1 = np.zeros((5,numrois))
    del ch
    for x in range(0,numrois):
        evoked_bands_pt1[0,x] = np.sum(tempout[delta,x])
        evoked_bands_pt1[1,x] = np.sum(tempout[theta,x])
        evoked_bands_pt1[2,x] = np.sum(tempout[alpha,x])
        evoked_bands_pt1[3,x] = np.sum(tempout[beta,x])
        evoked_bands_pt1[4,x] = np.sum(tempout[gamma,x])
# pt2 - single trials
    del tempout
    tempout = np.zeros((len(eppower),len(freqs),numrois))
    ch = 0
    tr = 0
    while tr<len(eppower):
        data = eppower[tr,:,:,:]
        for y in range(0,numrois):
            temp = data[y,:,:]
            for x in range(0,len(freqs)):
                tempout[tr,x,y]= np.sum(temp[x,167:500])
            del temp
        del data
        tr=tr+1
    evoked_bands_pt2 = np.zeros((len(eppower),5,numrois))
    for x in range(0,len(eppower)):
            data = tempout[x,:,:]
            for y in range(0,numrois):
                evoked_bands_pt2[x,0,y] = np.sum(data[delta,y])
                evoked_bands_pt2[x,1,y] = np.sum(data[theta,y])
                evoked_bands_pt2[x,2,y] = np.sum(data[alpha,y])
                evoked_bands_pt2[x,3,y] = np.sum(data[beta,y])
                evoked_bands_pt2[x,4,y] = np.sum(data[gamma,y])
      # this is what we've all been waiting for, get channel snr
    del tempout
    chSNR_bands_evoked = np.zeros((5,numrois))
    for x in range(0,numrois):
        temp1 = evoked_bands_pt1[:,x]
        temp2 = evoked_bands_pt2[:,:,x]
        chSNR_bands_evoked[0,x] = temp1[0]/stats.sem(temp2[:,0])
        chSNR_bands_evoked[1,x] = temp1[1]/stats.sem(temp2[:,1])
        chSNR_bands_evoked[2,x] = temp1[2]/stats.sem(temp2[:,2])
        chSNR_bands_evoked[3,x] = temp1[3]/stats.sem(temp2[:,3])
        chSNR_bands_evoked[4,x] = temp1[4]/stats.sem(temp2[:,4])
    del temp1 
    del temp2 
    del ch
    del tr
    del x
    del y  
# Induced power
# pt1 - average response
# create average response by first squaring the individual values
    ch = 0
    print('Estimating SNR for induced response per band')
    tempout = np.zeros((len(freqs),numrois))
    while ch<numrois:
        data = np.mean(np.square(eppower[:,ch,:,:]),0)
        for x in range(0,len(freqs)):
            temp = data[x,timewindow[0]:(timewindow[-1]+1)]
            temp = np.sum(temp)
            tempout[x,ch]=temp
            del temp
        ch=ch+1
        del data
    induced_bands_pt1 = np.zeros((5,numrois))
    del ch
    for x in range(0,numrois):
        induced_bands_pt1[0,x] = np.sum(tempout[delta,x])
        induced_bands_pt1[1,x] = np.sum(tempout[theta,x])
        induced_bands_pt1[2,x] = np.sum(tempout[alpha,x])
        induced_bands_pt1[3,x] = np.sum(tempout[beta,x])
        induced_bands_pt1[4,x] = np.sum(tempout[gamma,x])
# pt2 - single trials
      # this is what we've all been waiting for, get channel snr
    chSNR_bands_induced = np.zeros((5,numrois))
    for x in range(0,numrois):
        temp1 = induced_bands_pt1[:,x]
        temp2 = evoked_bands_pt2[:,:,x]
        chSNR_bands_induced[0,x] = temp1[0]/stats.sem(temp2[:,0])
        chSNR_bands_induced[1,x] = temp1[1]/stats.sem(temp2[:,1])
        chSNR_bands_induced[2,x] = temp1[2]/stats.sem(temp2[:,2])
        chSNR_bands_induced[3,x] = temp1[3]/stats.sem(temp2[:,3])
        chSNR_bands_induced[4,x] = temp1[4]/stats.sem(temp2[:,4])
    del temp1
    del temp2 
    del x

########################################################################################
#### Output    
    print('Complete. Returning output')
    return {'roi_snr_ASSR':chSNR_ASSR,'roi_snr_EVOKEDbands':chSNR_bands_evoked,
            'roi_snr_INDUCEDbands':chSNR_bands_induced} # create dictionary of outputs

### FUTURE DEVELOPMENT
##############################################################################################################
# Baseline signal to noise



    
def SNRpsdEPOCH(epoch_condition,starta,enda,snr_format,numrois,fstart,fend,
                snr_format_name,blchange,tmin,tmax,baseline,cond_events,cond_events_id,reject):
    
# SNR check for epoched MEG data using PSD estimation of frequency information.
# Inputs
# epoch_condition = data (raw or epoched, if raw event information will be used to create an epoched data set)
# starta = period of interest start (seconds)
# enda = period of interest end (seconds)
# snr_format = snr channel information (dict)
# numrois = number of roi channels (int)
# fstart = frequency window of interest start (int)
# fend = frequency window of interest end (int)   
# snr_format_name = name of channel type (e.g. ASSRnum - str)
# blchange = measure percent change from baseline (1 = yes, 2 = no) 
# tmin = event epoch minimum time (float)
# tmax = event epoch maximum time (float)
# baseline = mne baseline period (e.g (None,0),obj)
# cond_events = events for condition of interest
# cond_events_id = event ids for conditions to highlight cond of interest
# Outputs
# chSNR_ASSR (roiSNR) = snr for period of interest at each roi channel
# perchangeSNR = snr for percentage change between baseline and active period
   
    
    print('Importing additional modules')
    import scipy
    from scipy import stats
    from scipy import signal
    import numpy as np
    import copy
    import mne
    import matplotlib as mpl
    from matplotlib import mlab
################### PRE-ANALYSIS CHECKS
# Does the data require epochs creating?   
    check = np.shape(epoch_condition._data)
    check = np.size(check)
    if check<3:
        print('creating temporary epochs object')
        temp = mne.Epochs(epoch_condition,cond_events,cond_events_id,tmin,tmax,proj=False, #picks=picks,
                   baseline=baseline,preload=True,
                   reject=reject,add_eeg_ref=False)
        time = np.linspace(tmin,tmax,np.shape(temp._data)[2])
        epoch_condition = copy.deepcopy(temp)
        del temp
    else:
        time = np.linspace(tmin,tmax,np.shape(epoch_condition._data)[2]) 
# Create timewindow
    s = mlab.find(time==starta)
    e = mlab.find(time==enda)
    ee = e+1
    timewindow = np.arange(s,ee,1)
# Get sampling frequency
    fs = epoch_condition.info['sfreq']
######### PART 1 --- FRQ WINDOW OF INTEREST SNR
######### STEP 1 - PSD FROM EACH ROI CHANNEL (MEAN)
# psd is performed on the signal for the time window of interest (mean centred)
# in this version the appropriate window and overlap should be pre-determined
# for the resolution. future versions will automate this procedure to determine
# the windowing properties needed for a given resolution.
# Example - signal detrended and psd estimated for ~1hz resolution
#    f,Pxx = signal.welch((teste[timewindow]-mean(teste[timewindow])),
#                         fs=1000,nperseg = 1001,noverlap=np.round(1001/2), 
#                         detrend = 'linear',scaling='density')
    print('Extracting information from region of interest site')
    rois = np.zeros(numrois,dtype = np.int)
    for x in range (0,np.shape(rois)[0]): 
        text1 = snr_format_name
        text2 = str(x+1)
        text3 = text1+text2
        rois[x]=int(snr_format[text3])
    numfs = (fs/2)+1
    psdmatrix = np.zeros([numfs,numrois])
    print('Estimating grand average PSD for each ROI channel')
    for x in range(0,numrois):
        tempo = np.mean(epoch_condition._data[:,rois[x],timewindow],0)
        f,Pxx = signal.welch((tempo-np.mean(tempo)),fs=fs,nperseg=1001,
                             noverlap=np.round(1001/2),detrend ='linear',scaling='density')
        psdmatrix[:,x] = Pxx
    ff = np.round(f)
    frqwindow = np.arange((mlab.find(ff==fstart)),(mlab.find(ff==fend+1)),1)
######### STEP 2 - PSD FROM EACH ROI CHANNEL (SINGLE TRIALS)
    print('Estimating PSD for each ROI channel, single trials')
    psdmatrix2 = np.zeros([len(epoch_condition._data),(fs/2)+1,numrois])
    for x in range(0,numrois):
        data = epoch_condition._data[:,rois[x],timewindow]
        for y in range(0,len(epoch_condition._data)):
            tempo = data[y,:]
            f,Pxx = signal.welch((tempo-np.mean(tempo)),fs=fs,nperseg=1001,
                             noverlap=np.round(1001/2),detrend ='linear',scaling='density')
            psdmatrix2[y,:,x] = Pxx
######### STEP 3 - SNR 
    print('Estimating SNR')
    chSNR_ASSR = np.zeros(np.shape(rois)[0])
    for x in range(0,np.shape(rois)[0]):
        temp1 = psdmatrix[frqwindow,x]
        temp1 = np.sum(temp1)
        temp2 = np.sum(psdmatrix2[:,frqwindow,x],1)
        snr = temp1/stats.sem(temp2)
        chSNR_ASSR[x]=snr
        del temp1
        del temp2
        del snr
######### STEP 4 - if requested, snr from %signal change (baseline to window --- mean)
    if blchange == 1:
        print('Estimating SNR for percentage change between baseline and active period')
        bl = np.arange(0,(mlab.find(time==0)+1),1)
        bldpsd = np.zeros([(fs/2)+1,numrois])
        perchangeMean = np.zeros(numrois)
        bldpsdST = np.zeros([len(epoch_condition._data),(fs/2)+1,numrois])
        perchangeST = np.zeros([len(epoch_condition._data),numrois])
        percSNR = np.zeros(numrois)
        for x in range(0,numrois):
            bldat = np.mean(epoch_condition._data[:,rois[x],bl],0)
            f,Pxx = signal.welch((bldat-np.mean(bldat)),nfft=1001, nperseg = 500,noverlap = 250,fs=fs,detrend ='linear',scaling='density')
            bldpsd[:,x] = Pxx
        bldpsd = np.sum(bldpsd[frqwindow,:],0)
        for x in range(0,numrois):
            temp1 = np.sum(psdmatrix[frqwindow,x],0)
            perchangeMean[x] = ((temp1-bldpsd[x])/temp1)*100
                         
        for y in range(0,numrois):
            bldata = epoch_condition._data[:,rois[y],bl]
            for x in range(0,len(epoch_condition._data)):
                temp1 = bldata[x,:]
                f,Pxx = signal.welch((temp1-np.mean(temp1)),nfft=1001, nperseg = 500,noverlap = 250,fs=fs,detrend ='linear',scaling='density')
                bldpsdST[x,:,y] = Pxx
                temp1 = np.sum(psdmatrix2[x,frqwindow,y],0)
                perchangeST[x,y] = (temp1-(np.sum(bldpsdST[x,frqwindow,y],0))/temp1)*100
        
        for x in range(0,np.shape(rois)[0]):
            temp1 = perchangeMean[x]
            temp2 = perchangeST[x]
            snr = temp1/stats.sem(temp2)
            percSNR[x]=snr
        print('Finished')    
        return {'roiSNR':chSNR_ASSR,'perchangeSNR':percSNR}
    else:
        print('Finished')
        return chSNR_ASSR

    
def SNRpsdEPOCH_GRANDAV(epoch_condition,starta,enda,snr_format,fstart,fend,
                snr_format_name,tmin,tmax,baseline,cond_events,cond_events_id,reject):
    
    print('Importing additional modules')
    import scipy
    from scipy import stats
    from scipy import signal
    import numpy as np
    import copy
    import mne
    import matplotlib as mpl
    from matplotlib import mlab
    ################### PRE-ANALYSIS CHECKS
# Does the data require epochs creating?   
    check = np.shape(epoch_condition._data)
    check = np.size(check)
    if check<3:
        print('creating temporary epochs object')
        temp = mne.Epochs(epoch_condition,cond_events,cond_events_id,tmin,tmax,proj=False, #picks=picks,
                   baseline=baseline,preload=True,
                   reject=reject,add_eeg_ref=False)
        time = np.linspace(tmin,tmax,np.shape(temp._data)[2])
        epoch_condition = copy.deepcopy(temp)
        del temp
    else:
        time = np.linspace(tmin,tmax,np.shape(epoch_condition._data)[2]) 
# Create timewindow
    s = mlab.find(time==starta)
    e = mlab.find(time==enda)
    ee = e+1
    timewindow = np.arange(s,ee,1)
# Get sampling frequency
    fs = epoch_condition.info['sfreq']
    ######### PART 1 --- FRQ WINDOW OF INTEREST SNR
    ga = epoch_condition._data
    ga = ga[:,snr_format[snr_format_name],:]
    ga = np.mean(ga,0)
    ga = np.mean(ga,0)
    ga = ga-np.mean(ga)
    f,Pxx = signal.welch(ga[timewindow],fs=fs,nperseg=1001,
                             noverlap=np.round(1001/2),detrend ='linear',scaling='density')
    gapsd = Pxx
    del Pxx
    del ga
    ff = np.round(f)
    frqwindow = np.arange((mlab.find(ff==fstart)),(mlab.find(ff==fend+1)),1)
    delta  = np.arange((mlab.find(ff==1)),(mlab.find(ff==3+1)),1)
    theta = np.arange((mlab.find(ff==4)),(mlab.find(ff==7+1)),1)
    alpha = np.arange((mlab.find(ff==8)),(mlab.find(ff==14+1)),1)
    beta= np.arange((mlab.find(ff==15)),(mlab.find(ff==31+1)),1)
    gamma = np.arange((mlab.find(ff==32)),(mlab.find(ff==58+1)),1)
    ######### STEP 2 - PSD FROM EACH ROI CHANNEL (SINGLE TRIALS)
    print('Estimating PSD for each ROI channel, single trials')
    psdmatrix2 = np.zeros([len(epoch_condition._data),int((fs/2)+1)])
    for y in range(0,len(epoch_condition._data)):
            ga = epoch_condition._data
            ga = ga[y,snr_format[snr_format_name],:]
            ga = np.mean(ga,0)
            ga = ga-np.mean(ga)
            f,Pxx = signal.welch(ga[timewindow],fs=fs,nperseg=1001,
                             noverlap=np.round(1001/2),detrend ='linear',scaling='density')
            psdmatrix2[y,:] = Pxx
    ######### STEP 3 - SNR 
    print('Estimating SNR')
    temp1 = gapsd[frqwindow]
    temp1 = np.sum(temp1)
    temp2 = np.sum(psdmatrix2[:,frqwindow],1)
    snr = temp1/stats.sem(temp2)
    deltasnr = np.sum(gapsd[delta])/stats.sem(np.sum(psdmatrix2[:,delta],1))
    thetasnr = np.sum(gapsd[theta])/stats.sem(np.sum(psdmatrix2[:,theta],1))
    alphasnr = np.sum(gapsd[alpha])/stats.sem(np.sum(psdmatrix2[:,alpha],1))
    betasnr = np.sum(gapsd[beta])/stats.sem(np.sum(psdmatrix2[:,beta],1))
    gammasnr = np.sum(gapsd[gamma])/stats.sem(np.sum(psdmatrix2[:,gamma],1))
    
    return {'roisnrGA':snr,'deltasnr':deltasnr,'thetasnr':thetasnr,'alphasnr':alphasnr,
            'betasnr':betasnr,'gammasnr':gammasnr}
    

