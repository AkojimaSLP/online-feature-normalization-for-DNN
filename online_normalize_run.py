# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:37:43 2019

@author: a-kojima

add WPE

add online WPE

8/29
    debug

9/5 11 frames concat    

9/30 remove frame by frame WPE
    
"""

from beamformer import util
from beamformer import featnorm
import matplotlib.pyplot as pl

import numpy as np
import soundfile as sf
import time


#from maskestimator import feature



# ============================
# parame.
# ============================
SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 400
FFT_SHIFT = 160 
#CHANMEL_INDEX = [0, 2, 4, 6]
#CHANMEL_INDEX = [0, 4]

CHANMEL_INDEX = [0, 1,2,3,4,5,6,7]

USE_FIRST_FRAME = 10
PARAM_UPDATE_FREQ = 5
MIN_PARAM_UPDATE = 20 # 0.05 sec
SCM_GET_DELAY = 0


# ============================
# o,portant parame.
# ============================
ff_dropout= 0
recurrent_dropout = 0

SPEECH_AMP = 0.7
NOISE_AMP = 0.1

SELECT_CHANLE_INDEX = 0


# WPE param.
delay = 3
alpha = 0.99 #0.99
taps = 5 # num of frames for predict errow in next frames 10 (GOOGLE: 10)


def multi_channel_read(prefix=r'C:\Users\a-kojima\Documents\work_python\minatoku_go_bat\sample_data\20G_20GO010I_STR.CH{}.wav',
                       channel_index_vector=np.array([1, 2, 3, 4, 5, 6])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi



# ============================================
# make simulation data
# ============================================
multi_channels_data = sf.read(r'./data/sample_sp.wav')[0][:, CHANMEL_INDEX]
noise = sf.read(r'./data/noise_back.wav')[0][:, CHANMEL_INDEX]

# adjust size
min_size = np.min((np.shape(multi_channels_data)[0], np.shape(noise)[0]))
multi_channels_data = multi_channels_data[0:min_size, :]
noise = noise[0:min_size, :]

noise = noise / np.max(np.abs(noise)) * 0.2
multi_channels_data = multi_channels_data / np.max(np.abs(multi_channels_data)) * 0.7

noise_rand = np.random.normal(loc=0, scale=0.00001, size=(min_size, len(CHANMEL_INDEX)))
synth_r = noise + multi_channels_data + noise_rand

#single = synth_r[:, 0]
#
## ideal masl
## mask
#
#noise_for_mask = noise / np.max(np.abs(noise)) * 0.2
#multi_channels_data_for_mask = multi_channels_data / np.max(np.abs(multi_channels_data)) * 0.9
#
#complex_spectrum_noise, _ = util.get_3dim_spectrum_from_data(noise_for_mask, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
#complex_spectrum_speech, _ = util.get_3dim_spectrum_from_data(multi_channels_data_for_mask, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
#mask = np.abs(complex_spectrum_speech[0, :, :]) / (np.abs(complex_spectrum_speech[0, :, :]) + np.abs(complex_spectrum_noise[0, :, :]))
#
#

    

# ============================================
# cal complex spectrum
# ============================================
complex_spectrum, _ = util.get_3dim_spectrum_from_data(synth_r, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
number_of_frame = np.shape(complex_spectrum)[1] # C * T * F


# ============================================
# setup
# ============================================
normalizer = featnorm.Featnorm(feature_order=FFT_LENGTH // 2 + 1)

#==========================================
# get model
#==========================================
TRUNCATE_GRAD = 1
RECURRENT_INIT = 0.01
LR =  0.00001
NUMBER_OF_STACK = 11
    
# ==========================================================================
# para, for WPE
# ==========================================================================
channels = len(CHANMEL_INDEX)
sampling_rate = SAMPLING_FREQUENCY
frequency_bins = FFT_LENGTH // 2 + 1

Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
G = np.zeros((frequency_bins, channels * taps, channels))

# memory for save WPE
stack_complex_spec_for_wpe = []
stack_complex_spec_for_bf = []
number_of_dereverb_frames = 1 + delay + taps

# mmeory for overlapadd
stack_for_ola = []
frame = []
stack_complex_spec_target = []
#feature_extractor = feature.Feature(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT)

# ========================================
# frame by frame 
# ========================================
synth = synth_r[:, 0] * 0
st = 0
ed = st + FFT_LENGTH
number_of_update = 0
WINDOW = np.hanning(FFT_LENGTH)
predicted_mask = []
count_dereverb = 0 # check target frame
normalize_frame_stack = np.zeros((frequency_bins, number_of_frame))
input_frame_stack = np.zeros((frequency_bins, number_of_frame))
mean_stack = []
std_stack =[]

start_time = time.time()
SELECTED_CH = 0

for i in range(0, number_of_frame):      
    mag_spec = np.mean(np.abs(complex_spectrum[:, i, :]), axis=0)
    mean, std = normalizer.get_current_statistics(mag_spec)            
    normalized_frame = normalizer.get_normalize_frame(mean, std, mag_spec)            
    mean_stack = np.append(mean_stack, mean)
    std_stack.append(std)
    input_frame_stack[:, i] = mag_spec
    normalize_frame_stack[:, i] = normalized_frame


pl.figure()
pl.imshow(np.flipud(np.abs(normalize_frame_stack)), aspect='auto')
pl.title('input feature')

pl.figure()
pl.imshow(np.flipud(np.abs(input_frame_stack)), aspect='auto')
pl.title('normalize feature')

pl.show()

