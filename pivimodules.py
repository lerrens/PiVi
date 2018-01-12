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

# Set LED Row & Scale
bit = 16.0
bars_8 = np.array([0,1,2,3,4,9,18,36,64])
bars_8 = np.int_((bars_8/128)*(128))
bars_16 = np.array([0,1,2,3,4,5,6,7,8,9,18,27,36,48,72,96,128])
bars_16 = np.int_((bars_16/128)*(128))
maxdata = 2.0**(bit-1.0)-1.0



# Define: Light Up by case
def lightup_init(case, matrix_past, data, chunk, sample_rate):

	# SIXTEEN 16 BANDS

	if case == 1: # 16B_rb
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_16levels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 16 #(1-(y/H), H>15, better H<20)
		fgt = 0.3 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_16B_rb(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

	if case == 2: # 16B_gr
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_16levels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 30 #(1-(y/H), H>15, better H<20)
		fgt = 0.3 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_16B_gr(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

	if case == 3: # 16B_rb_H
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_16levels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 18 #(1-(y/H), H>15, better H<20)
		fgt = 0.95 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_16B_rb_hat(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

	# EIGHT 8 BANDS

	if case == 4: # 8B_rb
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_8levels(data, chunk, sample_rate, bars_8, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 16 #(1-(y/H), H>15, better H<20)
		fgt = 0.2 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_8B_rb(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

	if case == 5: # 8B_gr
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_8levels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 20 #(1-(y/H), H>15, better H<20)
		fgt = 0.15 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_8B_gr(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)



	if case == 6: # 8B_gr_H
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_8levels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 4
		colorHue = 18 #(1-(y/H), H>15, better H<20)
		fgt = 0.8 # *past
		# Convert Frequency Bands Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = fb2LED16B(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_8B_gr_hat(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)

	# TWO CHANNEL VU METERS	

	if case == -1: # 2ch_vu
		# Calculate Frequency Bands Amplitude 
		matrix = calculate_2chlevels(data, chunk, sample_rate, bars_16, maxdata)
		# Parameters
		dBSc = 2
		colorHue = 18 #(1-(y/H), H>15, better H<20)
		fgt = 0.2 # *past
		# Convert 2 Channel Amplitude to LED bar value
		matrix_past, matrix_LED, remain_LED = TC2LED(matrix_past, matrix, dBSc, fgt)
		# Show Unicorn LED
		lightup_2ch_vu(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt)
	return matrix_past
		
# HERE IS 16 SIXTEEN BANDS MODULES

# Define: 16 Frequency Bands Level Calculation
def calculate_16levels(data, chunk, sample_rate, bars_16, maxdata):
	# Convert raw data to 2 numpy array (L,R)
	data = unpack("%dh"%(len(data)/2),data)
	data = np.array(data, dtype='h')
	dataC = np.arange(np.int_(len(data)/2), dtype='h')
	for i in range (0,np.int_(len(data)/2)):
		dataC[i] = data[2*i]/2+data[2*i+1]/2

	# Apply FFT - real data so use rfft
	window = np.hanning(len(dataC)) 
	dataC = np.multiply(dataC,window) # Apply Window
	dataC = dataC[0:np.int_(len(dataC)/2)] # Samples cut: half
	fourier = np.fft.rfft(dataC)/len(dataC)
	fourier = fourier[0:np.int_(len(fourier))]
	fourier[1:np.int_(len(fourier))] = 2*fourier[1:np.int_(len(fourier))]

	# Remove last element in array to make it the same size as chunk
	fourier = np.delete(fourier,len(fourier)-1)

	# Calculate Frequency Bands Amplitude
	fbin = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

	for i in range (0,16):
		for j in range (bars_16[i],bars_16[i+1]):
			fbin[i]=fbin[i]+np.abs(fourier[j])
	matrix = 20*np.log10((fbin/maxdata)+0.00015)+3
	return matrix

# Define: Convert 16 Frequency Bands Amplitude to LED bar value (0-15)
def fb2LED16B(matrix_past, matrix, dBSc, fgt):
	matrix_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	remain_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	for i in range (0,16):
		remain_LED[i] = np.remainder((15*dBSc+matrix[i]),dBSc)
		for j in range (0,16):
			if (j*dBSc+matrix[i])>0:
				matrix_LED[i] = matrix_LED[i]+1
	matrix_past = np.int_(matrix_past*fgt + matrix_LED*(1-fgt))
	return matrix_past, matrix_LED, remain_LED

# CASE1: 16 Band Rainbow Lightup
def lightup_16B_rb(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 16):
		for y in range(0, 16):
			if y+1>(15-matrix_past[x]):
				if y==(15-matrix_past[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-x, y, x/15, 1-(y/colorHue), sat)
			else:
				unicornhathd.set_pixel_hsv(15-x, y, x/15, 1-(y/colorHue), 0.0)
	unicornhathd.show()
	return matrix_past

# CASE2: 16 Band GreenRed Lightup
def lightup_16B_gr(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 16):
		for y in range(0, 16):
			color = -0.15+y/30
			if color < 0:
				color = 0.0
			if y+1>(15-matrix_past[x]):
				if y==(15-matrix_past[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-x, y, color, 1-(y/colorHue), sat)
			else:
				unicornhathd.set_pixel_hsv(15-x, y, color, 1-(y/colorHue), 0.0)
	unicornhathd.show()
	return matrix_past

# CASE3: 16 Band Rainbow Lightup with hat
def lightup_16B_rb_hat(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 16):
		if matrix_past[x]<matrix_LED[x]:
			matrix_past[x]=matrix_LED[x]
		for y in range(0, 16):
			if y+1>(15-matrix_LED[x]):
				if y==(15-matrix_LED[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-x, y, x/15, 1-(y/colorHue), sat)
			else:
				unicornhathd.set_pixel_hsv(15-x, y, x/15, 1-(y/colorHue), 0.0)
			if y==(15-matrix_past[x]):
				unicornhathd.set_pixel_hsv(15-x, y-1, x/15, 0.0, 0.7)
	unicornhathd.show()
	return matrix_past


# HERE IS 8 EIGHT BANDS MODULES

# Define: 8 Frequency Bands Level Calculation
def calculate_8levels(data, chunk, sample_rate, bars_16, maxdata):
	# Convert raw data to 2 numpy array (L,R)
	data = unpack("%dh"%(len(data)/2),data)
	data = np.array(data, dtype='h')
	dataC = np.arange(np.int_(len(data)/2), dtype='h')
	for i in range (0,np.int_(len(data)/2)):
		dataC[i] = data[2*i]/2+data[2*i+1]/2

	# Apply FFT - real data so use rfft
	window = np.hanning(len(dataC)) 
	dataC = np.multiply(dataC,window) # Apply Window
	dataC = dataC[0:np.int_(len(dataC)/4)] # Samples cut: 1/4
	fourier = np.fft.rfft(dataC)/len(dataC)
	fourier = fourier[0:np.int_(len(fourier))]
	fourier[1:np.int_(len(fourier))] = 2*fourier[1:np.int_(len(fourier))]

	# Remove last element in array to make it the same size as chunk
	fourier = np.delete(fourier,len(fourier)-1)

	# Calculate Frequency Bands Amplitude
	fbin = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	for i in range (0,8):
		for j in range (bars_8[i],bars_8[i+1]):
			fbin[i]=fbin[i]+np.abs(fourier[j])
	matrix = 20*np.log10((fbin/maxdata)+0.00015)+6
	return matrix

# Define: Convert 8 Frequency Bands Amplitude to LED bar value (0-7)
def fb2LED8B(matrix_past, matrix, dBSc, fgt):
	matrix_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	remain_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	for i in range (0,16):
		remain_LED[i] = np.remainder((15*dBSc+matrix[i]),dBSc)
		for j in range (0,16):
			if (j*dBSc+matrix[i])>0:
				matrix_LED[i] = matrix_LED[i]+1
	matrix_past = np.int_(matrix_past*fgt + matrix_LED*(1-fgt))
	return matrix_past, matrix_LED, remain_LED

# CASE4: 8 Band Rainbow Lightup
def lightup_8B_rb(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 8):
		for y in range(0, 16):
			if y+1>(15-matrix_past[x]):
				if y==(15-matrix_past[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-2*x, y, 2*x/15, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, 2*x/15, 1-(y/colorHue), sat)
			else:
				unicornhathd.set_pixel_hsv(15-2*x, y, 2*x/15, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, 2*x/15, 1-(y/colorHue), 0.0)
	unicornhathd.show()
	return matrix_past

# CASE5: 8 Band GreenRed Lightup
def lightup_8B_gr(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 8):
		for y in range(0, 16):
			color = -0.15+y/30
			if color < 0:
				color = 0.0
			if y+1>(15-matrix_past[x]):
				if y==(15-matrix_past[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-2*x, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, color, 1-(y/colorHue), sat)
			else:

				unicornhathd.set_pixel_hsv(15-2*x, y, color , 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, color , 1-(y/colorHue), 0.0)
	unicornhathd.show()
	return matrix_past

# CASE6: 8 Band GreenRed Lightup with hat
def lightup_8B_gr_hat(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 8):
		if matrix_past[x]<matrix_LED[x]:
			matrix_past[x]=matrix_LED[x]
		for y in range(0, 16):
			# Bars
			color = -0.25+y/30
			if color < 0:
				color = 0.0
			if y+1>(15-matrix_LED[x]):
				if y==(15-matrix_LED[x]):
					#sat = remain_LED[x]*1.0/dBSc
					sat = 1.0
				else:	
					sat = 1.0
				unicornhathd.set_pixel_hsv(15-2*x, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, color, 1-(y/colorHue), sat)
			else:
				unicornhathd.set_pixel_hsv(15-2*x, y, color , 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-2*x-1, y, color , 1-(y/colorHue), 0.0)
			# Hats
			if y==(15-matrix_past[x]):
				unicornhathd.set_pixel_hsv(15-2*x, y-1, 2*x/15, 0.0, 0.7)
				unicornhathd.set_pixel_hsv(15-2*x-1, y-1, 2*x/15, 0.0, 0.7)
	unicornhathd.show()
	return matrix_past

# HERE IS 2 CHANNEL VU MODULES

# Define: 2 Channel Volume Level Calculation
def calculate_2chlevels(data, chunk, sample_rate, bars_16, maxdata):
	# Convert raw data to 2 numpy array (L,R)
	data = unpack("%dh"%(len(data)/2),data)
	data = np.array(data, dtype='h')
	dataL = np.arange(np.int_(len(data)/2), dtype='h')
	dataR = np.arange(np.int_(len(data)/2), dtype='h')
		
	for i in range (0,np.int_(len(data)/2)):
		dataL[i] = data[2*i]
		dataR[i] = data[2*i+1]
	dataL.astype('float_')
	dataR.astype('float_')
			
	# Scale raw RMS to dBFS
	Lrms = np.sqrt(np.average(np.power(dataL/maxdata,2)))
	Rrms = np.sqrt(np.average(np.power(dataR/maxdata,2)))
	avg = np.array([Lrms,Rrms])
	matrix = 20*np.log10(avg+0.00015)+3
	return matrix

# Define: Convert 2 Channel Volume Level to LED bar value (0-7)
def TC2LED(matrix_past, matrix, dBSc, fgt):
	matrix_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	remain_LED = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	for i in range (0,2):
		remain_LED[i] = np.remainder((15*dBSc+matrix[i]),dBSc)
		for j in range (0,16):
			if (j*dBSc+matrix[i])>0:
				matrix_LED[i] = matrix_LED[i]+1
	matrix_past = np.int_(matrix_past*fgt + matrix_LED*(1-fgt))
	return matrix_past, matrix_LED, remain_LED

# CASE-1: VU METER GreenRed Lightup
def lightup_2ch_vu(matrix_past, matrix_LED, remain_LED, dBSc, colorHue, fgt):
	for x in range(0, 2):
		for y in range(0, 16):
			color = -0.15+y/30
			if color < 0:
				color = 0.0
			if y+1>(15-matrix_past[x]):
				if y==(15-matrix_past[x]):
					sat = remain_LED[x]*1.0/dBSc
				else:	
					sat = 1.0
#				unicornhathd.set_pixel_hsv(15-8*x  , y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-1, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-2, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-3, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-4, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-5, y, color, 1-(y/colorHue), sat)
				unicornhathd.set_pixel_hsv(15-8*x-6, y, color, 1-(y/colorHue), sat)
#				unicornhathd.set_pixel_hsv(15-8*x-7, y, color, 1-(y/colorHue), sat)
			else:
#				unicornhathd.set_pixel_hsv(15-8*x  , y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-1, y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-2, y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-3, y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-4, y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-5, y, color, 1-(y/colorHue), 0.0)
				unicornhathd.set_pixel_hsv(15-8*x-6, y, color, 1-(y/colorHue), 0.0)
#				unicornhathd.set_pixel_hsv(15-8*x-7, y, color, 1-(y/colorHue), 0.0)
	unicornhathd.show()
	return matrix_past
