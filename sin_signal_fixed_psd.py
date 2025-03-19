import os
os.chdir('C:/Gibbs_SGVB_test/gibbs_sampling_sine_signal')
from scipy.stats import gamma, lognorm
import scipy.stats
import numpy as np
from data import true_var
import emcee
import matplotlib.pyplot as plt
from scipy import signal
import timeit
import glob

true_A = 2.0              # Amplitude
true_f = 5.0             # Frequency in Hz

n = 512

# Generate the true sine-wave signal
fs = 100                
dt = 1.0 / fs
t = np.linspace(0, (n - 1)*dt, n)
true_signal = true_A * np.sin(2 * np.pi * true_f * t)[:, np.newaxis]

# Generate VAR (2) time series as noise
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])

Simulation = true_var.VarmaSim(n=n)
freq = (np.arange(1,np.floor_divide(n, 2) + 1, 1) / (n))
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
for i in range(spec_true.shape[0]):
    for j in range(spec_true.shape[-1]):
        spec_true[i, j, j] = np.real(spec_true[i, j, j])
        
noise = Simulation.simData(varCoef, vmaCoef, sigma) 

# Create observed data as the sum of signal and noise
data = true_signal + noise

#plot periodogram for observed data
def periodogram(data, fs):
    n = len(data)
    
    ft_data = np.apply_along_axis(np.fft.fft, 0, data)
    if np.mod(n, 2) == 0:
        # n is even
        ft_data = ft_data[1:int(n/2 +1), :]
        freq = np.arange(1, int(n/2)+1) / n * fs
        
    else:
        # n is odd
        ft_data = ft_data[1:int((n-1)/2 +1), :]
        freq = np.arange(1, int((n-1)/2)+1) / n * fs

    periodograms = (np.abs(ft_data) ** 2 / n) / fs
    return periodograms, freq, ft_data

data_periodograms, freq, ft_data = periodogram(data, fs)


# estimate the parameters for signal given the fixed PSD
# set priors for two parameters
def log_prior(params):
    a = params[0]
    f = params[1]
    
    log_prior_a = gamma.logpdf(a, 3, scale=2)  
    log_prior_f = lognorm.logpdf(f, 0.5, scale=8)  
    
    #log_prior_a = uniform.logpdf(a, loc=1, scale=2)
    #log_prior_f = uniform.logpdf(f, loc=4, scale=2)
    
    return log_prior_a + log_prior_f

#Whittle likelihood
def log_likelihood(params):
    a, f = params
    signal = a * np.sin(2 * np.pi * f * t)[:, np.newaxis]
    
    ft_signal = np.fft.fft(signal, axis=0)
    
    if np.mod(n, 2) == 0:
        # n is even
        ft_signal = ft_signal[1:int(n/2 +1), :]
        
    else:
        # n is odd
        ft_signal = ft_signal[1:int((n-1)/2 +1), :]
        
    periodograms, freq, ft_data = periodogram(data, fs)
    residual_matrix = ft_data - ft_signal
    residual_matrix = residual_matrix[:, :, np.newaxis] #extend the shape to (512,2,1)
    inv_psd_matrices = np.linalg.pinv(spec_true / fs)
    
    likelihood_contributions = np.conj(np.transpose(residual_matrix, (0, 2, 1))) @ inv_psd_matrices @ residual_matrix
    
    log_likelihood = -0.5 * np.sum(likelihood_contributions.real)
    
    return log_likelihood


#create posterior
def log_probability(params):
    return log_prior(params) + log_likelihood(params)


# MCMC sampling
def metropolis_hastings(log_posterior, initial_params, proposal_scale, n_samples):
    
    samples = np.zeros((n_samples, len(initial_params)))
    current_params = np.array(initial_params)
    current_log_prob = log_posterior(current_params)
    accepted = 0
    
    for i in range(n_samples):
        proposed_params = current_params + np.random.normal(0, proposal_scale, size=len(initial_params))
        
        proposed_log_prob = log_posterior(proposed_params)
        acceptance_ratio = proposed_log_prob - current_log_prob

        if np.log(np.random.rand()) < acceptance_ratio:
            current_params = proposed_params
            current_log_prob = proposed_log_prob
            accepted += 1
        
        samples[i] = current_params

    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate

n_samples = 2000
initial_params = [3.0, 6.0] 
proposal_scale = [0.1, 0.2]

samples, acceptance_rate = metropolis_hastings(log_probability, initial_params, 
                                               proposal_scale, n_samples)

print(f"Acceptance Rate: {acceptance_rate:.2f}")

# plot the trace plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(samples[:, 0], label="a (Amplitude)")
axes[1].plot(samples[:, 1], label="f (Frequency)", color="orange")
axes[0].legend()
axes[1].legend()
plt.show()


# plot the hisogram
import matplotlib.pyplot as plt

a_samples = samples[:, 0]  # Amplitude (a)
f_samples = samples[:, 1]  # Frequency (f)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax[0].hist(a_samples, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax[0].set_title("Histogram of a")
ax[0].set_xlabel("a")
ax[0].set_ylabel("Frequency")

ax[1].hist(f_samples, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax[1].set_title("Histogram of f")
ax[1].set_xlabel("f")

plt.tight_layout()
plt.show()


























