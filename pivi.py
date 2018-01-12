#!/usr/bin/env python

import pivimodules2 as pv

# Sytem & ALSA
import alsaaudio as aa
import audioop
from time import sleep
from struct import unpack
import numpy as np
import sys 

# Unicorn HD
import colorsys
import math
import random
import time
from unicorn_hat_sim import unicornhathd as unicornhathd
unicornhathd.rotation(0)

# Set Audio Format
sample_rate = 44100
no_channels = 2
chunk = 512 # Use a multiple of 8
bit = 16.0

# Config Visualizer
case = -1


# Fetch Alsa Audio Data
#data_in = aa.PCM(aa.PCM_CAPTURE, aa.PCM_NORMAL)
data_in = aa.PCM(type=aa.PCM_CAPTURE,mode=aa.PCM_NORMAL,device="plug:alsa_monitor")
data_in.setchannels(no_channels)
data_in.setrate(sample_rate)
data_in.setformat(aa.PCM_FORMAT_S16_LE)
data_in.setperiodsize(chunk)


# Main Processing
print ("Enter Main Processing.....")
matrix_past = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
while True:
	# Read data from device	
	l,data = data_in.read()
	data_in.pause(1) # Pause capture whilst RPi processes data
	if l:
		# catch frame error
		try:
			# Calculate Frequency Bands Amplitude 
#			matrix = pv.calculate_levels(data, chunk, sample_rate, bars_16, maxdata)

			# Light up LED by case
			matrix_past = pv.lightup_init(case, matrix_past, data, chunk, sample_rate)

			# Convert Frequency Bands Amplitude to LED bar value
#			matrix_past, matrix_LED, remain_LED = pv.fb2LED16B(matrix_past, matrix, dBSc, fgt)

			# Show Unicorn LED
#			matrix_past = pv.lightup_16B_gr(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)
#			matrix_past = pv.lightup_16B_rb(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)
#			matrix_past = pv.lightup_16B_gr_hat(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

		except KeyboardInterrupt:
			unicornhathd.off()
#	sleep(0.5) #0.001
	sleep(0.001) #0.001
	data_in.pause(0) # Resume capture