#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:17:39 2023

@author: ayakabbara
"""

# A)Initialization
!pip install mne
!pip install mne_bids
!pip install pyprep
!pip install autoreject

#No need for these 6 lines if the mne_icalabel directory is in the directory
from google.colab import drive
drive.mount('/content/drive')
!gdown --id 1wXejh9N5MCry6i3auSU26IiUStYsl_4S
!unzip /content/drive/MyDrive/Colab/mne_icalabel.zip -d /content/mne_icalabel
!pip freeze
!pip install mne_icalabel


import mne_icalabel
import mne
from pyprep.find_noisy_channels import NoisyChannels
from mne_bids import read_raw_bids

# Set the path to the BIDS directory and the subject ID
from mne_bids import BIDSPath
from mne_bids.read import read_raw_bids
from mne.preprocessing import ICA
import sys

#The next line should be also updated depending on the location of mne_icalabel directory

sys.path.append('/content/mne_icalabel')
from mne_icalabel import label_components
from autoreject import AutoReject
from autoreject import get_rejection_threshold

#root_dir should be changed depending on the location if your data
root_dir='/content/dataHBN-BIDS'

# B) Functions defined
#Extracting epochs and baseline correction in task related eeg 
def epoching_task(raw,tmin,tmax,a,b): #default : tmin = -0.2 s , tmax = 0.8 , a = -0.2 , b = 0
    stim_channel = mne.pick_types(raw.info, meg=False, ref_meg=False, stim=True)
    if(len(stim_channel)!=0):
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw,events,tmin=tmin,tmax=tmax,baseline=(a,b),preload=True)
    else:
        annot = raw.annotations
        if(len(annot)!=0):
            events,event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw,events,tmin=tmin,tmax=tmax,baseline=(a,b),preload=True)
    return events , epochs 


#Epoching in resting state eeg
def epoching_rest(raw,length,overlap): #default length = 40 s , overlap = 0 (No overlap)
    epochs = mne.make_fixed_length_epochs(raw,preload=True, duration=length,overlap=overlap)
    return epochs

#Residual Bad Channel Detection:
def detectresbad(raw,chname,threshold):
    data = raw.get_data(picks=chname)
    rejected = list()
    for i in range(np.size(data,1)):
        if (np.std(data[:,i,:])>threshold):
            reject = raw.info['ch_names'][i]
            rejected.insert(len(rejected),reject)
            raw.info['bads'].insert(len(raw.info['bads']),reject)
                
    return raw, rejected

#Preprocessing:
def get_clean_epochs(subject_id, root_dir):

  #"""**Step 1: Load, filter, interpolate bad channels **"""

  # Specify the BIDS path using the BIDSPath object
  bids_path = BIDSPath(subject=subject_id, datatype='eeg', task='RestingState', root=root_dir)
  # Load the data into an MNE Raw object
  raw = read_raw_bids(bids_path)
  raw.load_data()
  # Filter data 
  raw.filter(1, 100)
  raw_copy = raw.copy()

  # Detect noisy channels using pyprep
  nd = NoisyChannels(raw, random_state=1337)
  nd.find_bad_by_ransac(channel_wise=True)

  for i in range(len(nd.bad_by_ransac)):
    raw.info['bads'].insert(len(raw.info['bads']), nd.bad_by_ransac[i])  
          
  # Interpolate noisy channels 
  raw.interpolate_bads(reset_bads=True)

  # Calculate the rejection rate
  total_channels = raw.info['nchan']  # Total number of channels
  bad_channels = len(raw.info['bads'])  # Number of rejected channels
  interpolation_rate = (bad_channels / total_channels) * 100

  # Re-reference to average 
  raw.set_eeg_reference('average', projection=True)
  raw.apply_proj()

  #"""**Step 2: Compute ICA **"""

  # Get the number of EEG channels
  n_eeg_channels = raw.info['nchan']

  ica = ICA(n_components=n_eeg_channels, random_state=42)
  ica.fit(raw)

  #"""**Step 3: Automatic IC artifactual detection**"""
  lab = label_components(raw, ica, method='iclabel')
  labels_list = lab['labels']
  indices_of_eye_blinks = [i for i in range(len(labels_list)) if labels_list[i] == 'eye blink']
  #"""**Step 4: Remove the component**"""
  if indices_of_eye_blinks:
    # Exclude IC components
    ica.exclude = indices_of_eye_blinks

    # Apply ICA to the raw data
    ica.apply(raw)

  # Get the total number of estimated ICs
  n_total_ics = ica.n_components_
  # Get the number of excluded ICs
  n_excluded_ics = len(ica.exclude)
  # Calculate the excluded rate
  excludedIC_rate = n_excluded_ics / n_total_ics
  # Filter data 
  raw.filter(1, 45)

  #"""**Step 5: Epoching**"""
  sampling_freq = raw.info['sfreq']  # Get the sampling frequency of your raw data
  duration_samples = 5000  # Replace with the duration in samples that you have

  # Convert duration from samples to seconds
  duration_seconds = duration_samples / sampling_freq

  # Create fixed-length epochs using the duration in seconds
  epochs = mne.make_fixed_length_epochs(raw, preload=True, duration=duration_seconds, overlap=0)

  # Print the number of epochs
  print(f"Number of epochs: {len(epochs)}")

  #"""**Step 6: Detect bad epochs and clean bad epochs using autoreject**"""

  ar = AutoReject()
  epochs_clean = ar.fit_transform(epochs)
  reject = get_rejection_threshold(epochs)

  # Calculate the rejection rate
  rejection_rate = (len(epochs) - len(epochs_clean)) / len(epochs) * 100

  # Calculate the three rejection rates
  channel_rejection_rate = (bad_channels / total_channels) * 100
  IC_rejection_rate = (n_excluded_ics / n_total_ics) * 100
  epoch_rejection_rate = rejection_rate

  # Return the epochs_clean and the three rejection rates
  return epochs_clean, channel_rejection_rate, IC_rejection_rate, epoch_rejection_rate

# C) Loop on subjects and call the preprocessing function
import os
# List all directories in the given directory
dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
# Get the last 4 characters of each directory name
subjects_id = [d[-4:] for d in dirs]

# Create the initial preprocessing directory if it doesn't exist
preprocessing_dir = os.path.join(root_dir, 'preprocessing')
if not os.path.exists(preprocessing_dir):
    os.mkdir(preprocessing_dir)

# Loop over the subject IDs and preprocess each subject's EEG data
for subject_id in subjects_id:
    # Preprocess the epochs for this subject
    epochs, channel_rejection, IC_rejection, epoch_rejection = get_clean_epochs(subject_id, root_dir)

    # Create a directory for this subject's epochs if it doesn't exist
    subject_dir = os.path.join(preprocessing_dir, subject_id)
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)

    # Save the epochs to a file
    epochs_file = os.path.join(subject_dir, 'clean_epochs.fif')
    epochs.save(epochs_file)

    # Create a note with the rejection rates
    note_file = os.path.join(subject_dir, 'rejection_rates.txt')
    with open(note_file, 'w') as f:
        f.write(f"Channel Rejection Rate: {channel_rejection}%\n")
        f.write(f"IC Rejection Rate: {IC_rejection}%\n")
        f.write(f"Epoch Rejection Rate: {epoch_rejection}%\n")