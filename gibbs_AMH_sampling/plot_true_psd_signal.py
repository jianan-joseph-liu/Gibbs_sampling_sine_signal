#import os
#os.chdir('D:/Gibbs_SGVB_test')
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import true_var


true_A = 2.0              # Amplitude
true_f = 5.0              # Frequency in Hz

n = 512

# Generate the true sine-wave signal
fs = 100                
dt = 1.0 / fs
t = np.linspace(0, (n - 1)*dt, n)
true_signal = true_A * np.sin(2 * np.pi * true_f * t)

# Generate VAR (2) time series as noise
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])

Simulation = true_var.VarmaSim(n=n)
noise = Simulation.simData(varCoef, vmaCoef, sigma)

# Generate true PSD
freq = (np.arange(1,np.floor_divide(n, 2) + 1, 1) / (n))
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
for i in range(spec_true.shape[0]):
    for j in range(spec_true.shape[-1]):
        spec_true[i, j, j] = np.real(spec_true[i, j, j])
        
spec_true = spec_true/fs

# Plot true psd and periodogram of signal
freq = freq*fs

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.subplots_adjust(hspace=0.08, wspace=0.08)

from matplotlib.lines import Line2D

for i in range(2):
    for j in range(2):
        if i == j:
            #plot true psd
            axes[i, j].plot(freq, np.log(spec_true[:,i,i]), linewidth=1, color = 'red', linestyle="-")
            
            #plot periodogram for noise
            f, Pxx_den0 = signal.periodogram(noise[:,i], fs=fs)
            f = f[1:]
            Pxx_den0 = Pxx_den0[1:]/2
            axes[i, j].plot(f, np.log(Pxx_den0), linestyle='-', color='lightgray', alpha=1)
            
            #plot periodogram for signal
            f, Period_signal = signal.periodogram(true_signal, fs=fs)
            f = f[1:]
            Period_signal = Period_signal[1:]/2
            axes[i, j].plot(f, np.log(Period_signal), linestyle='-', color='green', alpha=1)
            
            axes[i, j].text(0.95, 0.95, r'$\log(f_{{{}, {}}})$'.format(i+1, i+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)
            axes[i, j].grid(True)
            
        elif i < j:  
            
            axes[i, j].plot(freq, np.real(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-")
            
            y = np.apply_along_axis(np.fft.fft, 0, noise)
            if np.mod(n, 2) == 0:
                # n is even
                y = y[1:int(n/2 + 1)]
            else:
                # n is odd
                y = y[1:int((n-1)/2 + 1)]
            y = y / np.sqrt(n)
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            #plot noise periodogram f_ij
            axes[i, j].plot(f, np.real(cross_spectrum_fij)/fs,
                            linestyle='-', color='lightgray', alpha=1)
            
            axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)
        
            axes[i, j].grid(True)
            
        else:

            axes[i, j].plot(freq, np.imag(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-")
            
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            #plot noise periodogram f_ij
            axes[i, j].plot(f, np.imag(cross_spectrum_fij)/fs,
                            linestyle='-', color='lightgray', alpha=1)
            
            axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].grid(True)


























