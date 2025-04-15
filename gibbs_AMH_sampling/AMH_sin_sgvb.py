#import os
#os.chdir('D:/Gibbs_SGVB_test')

import h5py
import numpy as np
import pandas as pd
import inverse_spec_vi
from scipy.stats import gamma, lognorm
import true_var
import timeit


true_A = 3.0              # Amplitude
true_f = 5.0             # Frequency in Hz

n = 256

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
    ft_data = ft_data / np.sqrt(n * fs)
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
    covObj['cov'] = covObj['cov'] * (covObj['n']-2)/(covObj['n']-1) + dx.T @ dx / covObj['n']
    covObj['mean'] = X1 * (covObj['n']-1)/covObj['n'] + X / covObj['n']
    
    return covObj





# Initial parameters
initial_params = [3.5, 5.5]
d = len(initial_params)
Id = (0.1**2) * np.identity(d) / d
factor = (2.38**2) / d
beta = 0.05


# set the initial positions 
n_gibbs = 100       #number of iterations for Gibbs
current_signal = initial_params[0] * np.sin(2 * np.pi * initial_params[1] * t)[:, np.newaxis]
current_params = np.array(initial_params)
covObj = {'mean': current_params, 'cov': np.nan, 'n': 1}

accept_record = np.zeros(n_gibbs, dtype=int)
inverse_psd_all = np.zeros((n_gibbs, n//2, 2, 2), dtype=np.complex128)
gibbs_results = np.zeros((n_gibbs, len(initial_params)))
# Gibbs sampling
for gibbs_iter in range(n_gibbs):
    residuals = data - current_signal
    print('Current Gibbs iteration setp is: ', gibbs_iter)
    Spec = inverse_spec_vi.SpecVI(residuals)
    result_list = Spec.runModel(N_delta=32, N_theta=32, lr_map=0.012, 
                                ntrain_map=1500, sparse_op=False)
#    est_psd = result_list[0] / fs
    
    inverse_psd = result_list[0] * fs

    inverse_psd_all[gibbs_iter] = inverse_psd
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
            
        ft_signal = ft_signal / np.sqrt(n * fs)
            
        periodograms, freq, ft_data = periodogram(data, fs)
        #df = freq[1] - freq[0]
        
        residual_matrix = ft_data - ft_signal
        residual_matrix = residual_matrix[:, :, np.newaxis]
        inv_psd_matrices = inverse_psd
        
        likelihood_contributions = np.conj(np.transpose(residual_matrix, (0, 2, 1))) @ inv_psd_matrices @ residual_matrix
        
        log_likelihood = -np.sum(likelihood_contributions.real)
        
        return log_likelihood

    # posterior
    def log_posterior(params):
        return log_prior(params) + log_prob(params)


    # Adaptive MCMC sampling
    if np.random.rand() < beta or gibbs_iter <= 2*d:
        proposed_params = np.random.multivariate_normal(current_params, Id)
    else:
        proposed_params = np.random.multivariate_normal(current_params, factor * covObj['cov'])
        print('factor * cov = ', factor * covObj['cov'])

    current_log_prob = log_posterior(current_params)
    proposed_log_prob = log_posterior(proposed_params)

    acceptance_ratio = proposed_log_prob - current_log_prob

    if np.log(np.random.rand()) < acceptance_ratio:
        current_params = proposed_params
        accept_record[gibbs_iter] = 1

    print('a_est = ', current_params[0])
    print('f_est = ', current_params[1])

    mean_accept_rate = np.mean(accept_record[:gibbs_iter + 1])
    print(f"Current Gibbs iteration: {gibbs_iter}, mean accept rate: {mean_accept_rate:.3f}")
    
    # Save parameters from Gibbs iteration
    gibbs_results[gibbs_iter, :] = current_params

    # Update covariance object for adaptive MH
    covObj = updateCov(X=current_params, covObj=covObj)

    current_signal = current_params[0] * np.sin(2 * np.pi * current_params[1] * t)[:, np.newaxis]

    

''' 
df = pd.DataFrame(gibbs_results, columns=["a_est", "f_est"])
csv_filename = "gibbs_sampling_results_6.csv"
df.to_csv(csv_filename, index=False)


with h5py.File("gibbs_inverse_psd_results_6.h5", "w") as f:
    f.create_dataset("psd", data=inverse_psd_all)
'''    
    
    




'''
def updateCov(X, covObj=np.nan):
    if not isinstance(covObj, dict): 
        covObj = {'mean':X, 'cov':np.nan, 'n':1}
        return covObj
    
    covObj['n'] += 1
    X1 = covObj['mean']
    
    if covObj['n'] == 2:
        covObj['mean'] = X1/2 + X/2
        dX1 = X1 - covObj['mean']
        dX2 = X - covObj['mean']
        covObj['cov'] = dX1**2 + dX2**2
        return covObj
    
    dx = X - X1
    covObj['mean'] = X1 * (covObj['n']-1)/covObj['n'] + X / covObj['n']
    covObj['cov'] = covObj['cov'] * (covObj['n']-2)/(covObj['n']-1) + \
                    dx*(X - covObj['mean']) / (covObj['n'] - 1)
    
    return covObj
'''























    
 
    