import os
os.chdir('C:/Gibbs_SGVB_test/gibbs_sampling_sine_signal')
import spec_vi_v2
from scipy.stats import gamma, lognorm
import numpy as np
from data import true_var
import matplotlib.pyplot as plt
from scipy import signal
import timeit

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

#periodogram for observed data
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
# function for trace plots
def plot_trace(samples, labels=["a (Amplitude)", "f (Frequency)"]):
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 6), sharex=True)

    for i in range(n_params):
        axes[i].plot(samples[:, i], label=labels[i])
        axes[i].legend()
        axes[i].set_ylabel(labels[i])
        
    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.show()

# function for histogram plots
def plot_histograms(samples, labels=["a (Amplitude)", "f (Frequency)"], bins=50):
   
    n_params = samples.shape[1]
    fig, ax = plt.subplots(1, n_params, figsize=(12, 5), sharey=True)

    for i in range(n_params):
        ax[i].hist(samples[:, i], bins=bins, alpha=0.7)
        ax[i].set_title(f"Histogram of {labels[i]}")
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel("Frequency" if i == 0 else "")

    plt.tight_layout()
    plt.show()


# set priors for two parameters
def log_prior(params):
    a = params[0]
    f = params[1]
    
    log_prior_a = gamma.logpdf(a, 3, scale=2)  
    log_prior_f = lognorm.logpdf(f, 0.5, scale=8)  
    
    #log_prior_a = uniform.logpdf(a, loc=1, scale=2)
    #log_prior_f = uniform.logpdf(f, loc=4, scale=2)
    
    return log_prior_a + log_prior_f


# set the initial positions and hyperparameters
n_gibbs = 5       #number of iterations for Gibbs
n_samples = 2000  #number of iterations for MCMC
initial_params = [3.2, 6.3]
proposal_scale = [0.1, 0.2]
current_signal = 3 * np.sin(2 * np.pi * 6 * t)[:, np.newaxis]
burn_in = 500


# Gibbs sampling
for gibbs_iter in range(n_gibbs):
    residuals = data - current_signal
    print('Current Gibbs iteration setp is: ', gibbs_iter)
    Spec = spec_vi_v2.SpecVI(residuals)
    result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=0.012, 
                                ntrain_map=3000, sparse_op=False)
    est_psd = result_list[0] / fs


    #Whittle likelihood
    def log_prob(params):
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
        residual_matrix = residual_matrix[:, :, np.newaxis]
        inv_psd_matrices = np.linalg.pinv(est_psd)
        
        likelihood_contributions = np.conj(np.transpose(residual_matrix, (0, 2, 1))) @ inv_psd_matrices @ residual_matrix
        
        log_likelihood = -0.5 * np.sum(likelihood_contributions.real)
        
        return log_likelihood

    # posterior
    def log_posterior(params):
        return log_prior(params) + log_prob(params)


    # MCMC sampling
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
    print(f"For Gibbs iteration {gibbs_iter+1}, the acceptance rate is {acceptance_rate:.2f}")

    plot_trace(samples)
    plot_histograms(samples)

    # discard the burn in samples and create updated sine signal wave
    post_burn_in_samples  = samples[burn_in:, :]
    
    a_est = np.median(post_burn_in_samples[:,0])
    f_est = np.median(post_burn_in_samples[:,1])

    current_signal = a_est * np.sin(2 * np.pi * f_est * t)[:, np.newaxis]

    # plot estimated psd for noise and estimated periodogram for signal
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    from matplotlib.lines import Line2D

    for i in range(2):
        for j in range(2):
            if i == j:
                f, Pxx_den0 = signal.periodogram(data[:,i], fs=fs)
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:] 
                
                #plot data periodogram
                axes[i, j].plot(f, np.log(Pxx_den0), marker='', markersize=0, linestyle='-', color='lightgray', alpha=1)
                #plot estimated psd
                axes[i, j].plot(freq, np.log(est_psd[..., i, i]), linewidth=1, color='blue', linestyle="-")
                #plot true psd
                axes[i, j].plot(freq, np.log(spec_true[:,i,i]/fs), linewidth=1, color = 'red', linestyle="-.")
                #plot estimated signal periogoram
                f, Pxx_den1 = signal.periodogram(np.squeeze(current_signal), fs=fs)
                f = f[1:]
                Pxx_den1 = Pxx_den1[1:]
                axes[i, j].plot(f, np.log(Pxx_den1), marker='', markersize=0, linestyle='-.', color='green', alpha=1)
                #plot true signal periogoram
                f, Pxx_den2 = signal.periodogram(np.squeeze(true_signal), fs=fs)
                f = f[1:]
                Pxx_den2 = Pxx_den2[1:]
                axes[i, j].plot(f, np.log(Pxx_den2), marker='', markersize=0, linestyle='-', color='green', alpha=1)
                
                axes[i, j].text(0.95, 0.95, r'$\log(f_{{{}, {}}})$'.format(i+1, i+1), transform=axes[i, j].transAxes, 
                                horizontalalignment='right', verticalalignment='top', fontsize=14)
                
                axes[i, j].grid(True)
                
            elif i < j:  
                
                y = np.apply_along_axis(np.fft.fft, 0, noise)
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0:int(n/2)]
                else:
                    # n is odd
                    y = y[0:int((n-1)/2)]
                y = y / np.sqrt(n)
                #y = y[1:]
                cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
                
                #plot data periodogram f_ij
                axes[i, j].plot(f, np.real(cross_spectrum_fij)/fs,
                                marker='', markersize=0, linestyle='-', color='lightgray', alpha=1)
                #plot estimated psd
                axes[i, j].plot(freq, np.real(est_psd[..., i, j]), linewidth=1, color='blue', linestyle="-")
                #plot true psd
                axes[i, j].plot(freq, np.real(spec_true[:,i,j])/fs, linewidth=1, color = 'red', linestyle="-.")
                
                axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                                horizontalalignment='right', verticalalignment='top', fontsize=14)
            
                axes[i, j].grid(True)
                
            else:

                y = np.apply_along_axis(np.fft.fft, 0, noise)
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0:int(n/2)]
                else:
                    # n is odd
                    y = y[0:int((n-1)/2)]
                y = y / np.sqrt(n)
                #y = y[1:]
                cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
                
                #plot data periodogram f_ij
                axes[i, j].plot(f, np.imag(cross_spectrum_fij)/fs,
                                marker='', markersize=0, linestyle='-', color='lightgray', alpha=1)
                #plot estimated psd
                axes[i, j].plot(freq, np.imag(est_psd[..., i, j]), linewidth=1, color='blue', linestyle="-")
                #plot true psd
                axes[i, j].plot(freq, np.imag(spec_true[:,i,j])/fs, linewidth=1, color = 'red', linestyle="-.")
             
                
                axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                                horizontalalignment='right', verticalalignment='top', fontsize=14)

                axes[i, j].grid(True)
    fig.text(0.5, 0.08, 'Frequency', ha='center', va='center', fontsize=20)
    fig.text(0.08, 0.5, 'Spectral Densities', ha='center', va='center', rotation='vertical', fontsize=20)

    fig.legend(handles=[Line2D([], [], color='lightgray', label='Data Periodogram'),
                    Line2D([], [], color='red', label='True PSD'),
                    Line2D([], [], color='blue', label='Estimated PSD'),
                    Line2D([], [], color='green', linestyle='-.', label='Estimated Signal Periodogram'),
                    Line2D([], [], color='green', linestyle='-', label='True Signal Periodogram')],
                 loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=14)
    
    plt.suptitle(f"PSD Estimation (Gibbs Iteration {gibbs_iter+1})", y=1)
    plt.tight_layout()
    plt.savefig(f"psd_plot_itr{gibbs_iter+1:02d}.png")
    plt.show()
    plt.close() 























