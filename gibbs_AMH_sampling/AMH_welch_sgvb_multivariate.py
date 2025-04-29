import os
os.chdir('D:/Gibbs_SGVB_test')
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import inverse_spec_vi
from scipy.stats import uniform
import true_var
from matplotlib.lines import Line2D

n = 1024
fs = 100
dt = 1.0 / fs

# Generate the true sine-wave signal
true_A = 0.1              # Amplitude
true_f = 5.0             # Frequency in Hz
t = np.linspace(0, (n - 1)*dt, n)
true_signal = true_A * np.sin(2 * np.pi * true_f * t)[:, np.newaxis]

# create one noise signal and observed dataset
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]],
                    [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1., 0.], [0., 1.]]])

Simulation = true_var.VarmaSim(n=n)
noise = Simulation.simData(varCoef, vmaCoef, sigma)
data = true_signal + noise


# create mutiple noise signals
n_realizations = 500
noise_array = np.zeros((n_realizations, n, 2))

for i in range(n_realizations):
    Simulation = true_var.VarmaSim(n=n)
    noise = Simulation.simData(varCoef, vmaCoef, sigma)
    noise_array[i] = noise

p = noise_array.shape[-1]

# Function that computes the periodogram
def get_periodogram(x, fs, **kwargs):
    n, p = x.shape
    periodogram = np.zeros((n // 2, p, p), dtype=complex)
    for row_i in range(p):
        for col_j in range(p):
            if row_i == col_j:
                f, Pxx_den0 = signal.periodogram(
                    x[:, row_i], fs=fs, scaling="density"
                )
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:]
                periodogram[..., row_i, col_j] = Pxx_den0 / 2
            else:

                y = np.apply_along_axis(np.fft.fft, 0, x)
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0 : int(n / 2)]
                else:
                    # n is odd
                    y = y[0 : int((n - 1) / 2)]
                y = y / np.sqrt(n)
                cross_spectrum_fij = y[:, row_i] * np.conj(y[:, col_j])
                cross_spectrum_fij = cross_spectrum_fij / fs
                periodogram[..., row_i, col_j] = cross_spectrum_fij

    return periodogram, f


# Compute periodogram for all noise series
periodograms_all = np.zeros((n_realizations, n // 2, p, p), dtype=complex)

for i in range(n_realizations):
    x = noise_array[i]  # shape (n, 2)
    period, f = get_periodogram(x, fs=fs)
    periodograms_all[i] = period

# Compute the median periodogram
median_welch = np.zeros((n // 2, 2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        if i == j:
            median_welch[:, i, j] = np.mean(periodograms_all[:, :, i, j].real, axis=0)
        else:
            real_part = np.mean(periodograms_all[:, :, i, j].real, axis=0)
            imag_part = np.mean(periodograms_all[:, :, i, j].imag, axis=0)
            median_welch[:, i, j] = real_part + 1j * imag_part


# create FFT function
def fft(data, fs):
    n = len(data)
    
    ft_data = np.apply_along_axis(np.fft.fft, 0, data)
    if np.mod(n, 2) == 0:
        # n is even
        ft_data = ft_data[1:int(n/2 +1), :] / np.sqrt(n * fs)
        freq = np.arange(1, int(n/2)+1) / n * fs
        
    else:
        # n is odd
        ft_data = ft_data[1:int((n-1)/2 +1), :] / np.sqrt(n * fs)
        freq = np.arange(1, int((n-1)/2)+1) / n * fs

    return freq, ft_data

# set priors for two parameters
A_RANGE = (0.01, 1.0)
F0_RANGE = (3.0, 7.0)
def log_prior(params):
    a = params[0]
    f = params[1]
    
    #log_prior_a = gamma.logpdf(a, 1, scale=2)  
    #log_prior_f = lognorm.logpdf(f, 0.1, scale=4)  
    
    log_prior_a = uniform.logpdf(a, loc=A_RANGE[0], scale=A_RANGE[1] - A_RANGE[0])
    log_prior_f = uniform.logpdf(f, loc=F0_RANGE[0], scale=F0_RANGE[1] - F0_RANGE[0])
    
    return log_prior_a + log_prior_f


#Whittle likelihood
def log_prob(params, inv_psd_matrices):
    a, f = params
    signal = a * np.sin(2 * np.pi * f * t)[:, np.newaxis]
        
    freq, ft_signal = fft(signal, fs)
    freq, ft_data = fft(data, fs)
    
    residual_matrix = (ft_data - ft_signal)[:, :, np.newaxis]
    
    likelihood_contributions = np.conj(np.transpose(residual_matrix, (0, 2, 1))) @ inv_psd_matrices @ residual_matrix
    
    log_likelihood = -np.sum(likelihood_contributions.real)
    return log_likelihood


# posterior
def log_posterior(params, inv_psd_matrices):
    log_prior_val = log_prior(params)
    log_likelihood_val = log_prob(params, inv_psd_matrices)
    return log_prior_val + log_likelihood_val



# function for AMH
def updateCov(X, covObj=np.nan):
    if not isinstance(covObj, dict): 
        covObj = {'mean':X, 'cov':np.nan, 'n':1}
        return covObj
    
    covObj['n'] += 1
    X1 = np.asmatrix(covObj['mean'])
    X  = np.asmatrix(X)
    
    if covObj['n'] == 2:
        covObj['mean'] = X1/2 + X/2
        dX1 = X1 - covObj['mean']
        dX2 = X - covObj['mean']
        covObj['cov'] = dX1.T @ dX1 + dX2.T @ dX2
        return covObj
    
    dx = X1 - X
    covObj['cov'] = covObj['cov'] * (covObj['n']-2)/(covObj['n']-1) + dx.T @ dx / (covObj['n'])
    covObj['mean'] = X1 * (covObj['n']-1)/covObj['n'] + X / covObj['n']
    
    return covObj


# set the initial positions
initial_params = [np.random.uniform(*A_RANGE), np.random.uniform(*F0_RANGE)]
d = len(initial_params)
Id = (0.5**2) * np.identity(d)/2
factor = 1#(2.38**2) / d
beta = 0.05

n_gibbs = 30000       #number of iterations for Gibbs
current_signal = initial_params[0] * np.sin(2 * np.pi * initial_params[1] * t)[:, np.newaxis]
current_params = np.array(initial_params)
covObj = {'mean': current_params, 'cov': np.nan, 'n': 1}


# empty set to recall all the information
accept_record = np.zeros(n_gibbs, dtype=int)
accept_rate_record = np.zeros(n_gibbs)
#inverse_psd_all = np.zeros((n_gibbs, n//2, 2, 2), dtype=np.complex128)
inverse_psd_all = []
gibbs_results = np.zeros((n_gibbs, len(initial_params)))


# Gibbs sampling
inverse_median_welch = np.linalg.inv(median_welch)
residuals = data - current_signal

for gibbs_iter in range(n_gibbs):
    print('Current Gibbs iteration setp is: ', gibbs_iter)
    
    if gibbs_iter < 15000:
        inv_psd_matrices = inverse_median_welch
        
    elif gibbs_iter % 2000 == 0:
        residuals = data - current_signal
        Spec = inverse_spec_vi.SpecVI(residuals)
        result_list = Spec.runModel(N_delta=32, N_theta=32, lr_map=0.012, 
                                    ntrain_map=1500, sparse_op=False)
        
        inv_psd_matrices = result_list[0] * fs
        inverse_psd_all.append(inv_psd_matrices)
        
    # Adaptive Metropolis step
    if np.random.rand() < beta or gibbs_iter <= 3000:
        proposed_params = np.random.multivariate_normal(current_params, Id)
    else:
        proposed_params = np.random.multivariate_normal(current_params, factor * covObj['cov'])
        
    current_log_post = log_posterior(current_params, inv_psd_matrices)
    proposed_log_post = log_posterior(proposed_params, inv_psd_matrices)

    acceptance_ratio = proposed_log_post - current_log_post

    if np.log(np.random.rand()) < acceptance_ratio:
        current_params = proposed_params
        accept_record[gibbs_iter] = 1

    print('a_est = ', current_params[0])
    print('f_est = ', current_params[1])

    mean_accept_rate = np.mean(accept_record[:gibbs_iter + 1])
    accept_rate_record[gibbs_iter] = mean_accept_rate
    print(f"Current Gibbs iteration: {gibbs_iter}, mean accept rate: {mean_accept_rate:.3f}")
    
    # Save parameters from Gibbs iteration
    gibbs_results[gibbs_iter, :] = current_params

    # Update covariance object for adaptive MH
    covObj = updateCov(X=current_params, covObj=covObj)

    # Update current_signal for residuals
    current_signal = current_params[0] * np.sin(2 * np.pi * current_params[1] * t)[:, np.newaxis]

inverse_psd_all = np.array(inverse_psd_all)


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# plots for gibbs_results, accept_rate_record, inverse_psd_all
# trace plots
print('median a = ', np.median(gibbs_results[:, 0]))
print('median f = ', np.median(gibbs_results[:, 1]))

a_est = gibbs_results[3000:, 0]
f_est = gibbs_results[3000:, 1]

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(a_est, label="Amplitude", color="blue", alpha=0.7)
ax[0].axhline(y=true_A, color="red", linestyle="--", label="True Value")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Trace Plot of Amplitude")
ax[0].legend()

ax[1].plot(f_est, label="Frequency", color="blue", alpha=0.7)
ax[1].axhline(y=true_f, color="red", linestyle="--", label="True Value")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Trace Plot of Frequency")
ax[1].legend()

ax[1].set_xlabel("Iteration")
plt.tight_layout()
plt.show()


# find quantiles for estimated psd
psd_all = np.linalg.inv(inverse_psd_all)

p_dim = psd_all.shape[-1]
num_freq = psd_all.shape[1]
spectral_density_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

diag_indices = np.diag_indices(p_dim)
spectral_density_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]), [0.025, 0.5, 0.975], axis=0)

triu_indices = np.triu_indices(p_dim, k=1)
real_part = (np.real(psd_all[:, :, triu_indices[1], triu_indices[0]]))
imag_part = (np.imag(psd_all[:, :, triu_indices[1], triu_indices[0]]))

spectral_density_q[0, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.025, axis=0) + 1j * np.quantile(imag_part, 0.025, axis=0)).T
spectral_density_q[1, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.50, axis=0) + 1j * np.quantile(imag_part, 0.50, axis=0)).T
spectral_density_q[2, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.975, axis=0) + 1j * np.quantile(imag_part, 0.975, axis=0)).T

spectral_density_q[:, :, triu_indices[0], triu_indices[1]] = np.conj(spectral_density_q[:, :, triu_indices[1], triu_indices[0]])

freq = (np.arange(1,np.floor_divide(n, 2) + 1, 1) / (n))
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
for i in range(spec_true.shape[0]):
    for j in range(spec_true.shape[-1]):
        spec_true[i, j, j] = np.real(spec_true[i, j, j])
spec_true = spec_true / fs
freq = freq * fs

spec_mat_lower, spec_mat_median, spec_mat_upper = spectral_density_q


# find quantiles for estmated signals' periodogram
signal_periodograms = np.zeros((len(a_est), n//2))

for i in range(len(a_est)):
    signal_i  = a_est[i] * np.sin(2 * np.pi * f_est[i] * t)
    f, Pxx = signal.periodogram(signal_i, fs=fs)
    Pxx = Pxx[1:]/2
    signal_periodograms[i] = Pxx

ci_lower = np.percentile(signal_periodograms, 2.5, axis=0)
median = np.percentile(signal_periodograms, 50, axis=0)
ci_upper = np.percentile(signal_periodograms, 97.5, axis=0)

signal_summary = np.stack([ci_lower, median, ci_upper], axis=0)


# plot the credible interval for estimated psd
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(2):
    for j in range(2):
        if i == j:
            f, Pxx_den0 = signal.periodogram(noise[:,i], fs=fs)
            f = f[1:]
            Pxx_den0 = Pxx_den0[1:]/2
            
            #plot data periodogram
            axes[i, j].plot(f, np.log(Pxx_den0), marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
            
            #plot median of estimated psd
            axes[i, j].plot(freq, np.log(np.real(spec_mat_median[..., i, i])), linewidth=1, color='blue', linestyle="-")
            
            #plot true psd
            axes[i, j].plot(freq, np.log(spec_true[:,i,i]), linewidth=1, color = 'red', linestyle="-.")
            
            #fill the credible interval
            axes[i, j].fill_between(freq, np.log(np.real(spec_mat_lower[..., i, i])),
                            np.log(np.real(spec_mat_upper[..., i, i])), color='lightblue', alpha=1)
            
            #plot true signal periogoram
            f, Pxx_den2 = signal.periodogram(np.squeeze(true_signal), fs=fs)
            f = f[1:]
            Pxx_den2 = Pxx_den2[1:]/2
            axes[i, j].plot(f, np.log(Pxx_den2), marker='', markersize=0, linestyle='-', color='red', alpha=1)
            
            # plot median and CI of estimated signal
            axes[i, j].plot(freq, np.log(signal_summary[1]), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, np.log(signal_summary[0]),
                            np.log(signal_summary[2]), color='lightgreen', alpha=1)
            
            
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
            
            #plot noise periodogram f_ij
            axes[i, j].plot(f, np.real(cross_spectrum_fij)/fs,
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
            
            #plot median of estimated psd
            axes[i, j].plot(freq, np.real(spec_mat_median[..., i, j]), linewidth=1, color='blue', linestyle="-")
            
            #fill the credible interval
            axes[i, j].fill_between(freq, np.real(spec_mat_lower[..., i, j]),
                            np.real(spec_mat_upper[..., i, j]), color='lightblue', alpha=1)
            
            #plot true psd
            axes[i, j].plot(freq, np.real(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-.")
            
            axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)
        
            axes[i, j].grid(True)
            
        else:

            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            #plot data periodogram f_ij
            axes[i, j].plot(f, np.imag(cross_spectrum_fij)/fs,
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
            
            #plot median of estimated psd
            axes[i, j].plot(freq, np.imag(spec_mat_median[..., i, j]), linewidth=1, color='blue', linestyle="-")
            
            #fill the credible interval
            axes[i, j].fill_between(freq, np.imag(spec_mat_lower[..., i, j]),
                            np.imag(spec_mat_upper[..., i, j]), color='lightblue', alpha=1)
            
            #plot true psd
            axes[i, j].plot(freq, np.imag(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-.")
         
            
            axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].grid(True)
fig.text(0.5, 0.08, 'Frequency', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, 'Spectral Densities', ha='center', va='center', rotation='vertical', fontsize=20)

fig.legend(handles=[Line2D([], [], color='lightgray', label='Noise Periodogram'),
                Line2D([], [], color='red', label='True PSD'),
                Line2D([], [], color='blue', label='Estimated PSD'),
                Line2D([], [], color='lightblue', label='95% credible interval')],
             loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=14)


# plot the acceptance rate
accept_ = accept_rate_record[3000:,]

plt.figure(figsize=(10,6))
plt.plot(np.arange(len(accept_)), accept_, label='Acceptance Rate')
plt.xlabel('Gibbs Iteration')
plt.ylabel('Cumulative Acceptance Rate')
plt.title('Acceptance Rate Over Gibbs Sampling')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


































