import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.io import loadmat, savemat
import signal_process as sp
from Functional_Connectivity import *
from Kuramoto import *
from itertools import chain
from scipy.signal import butter, sosfilt, welch
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.gradient.alignment import procrustes_alignment
from scipy import stats

data =  loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")
    
# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
reference10 = reference[:,0:10]


data_rs = data['TS_RS'][0]
data_comp = data['TS_COMP'][0]


num_points = 93
num_subjects = 100
num_nodes = 1000

time_window_fcd  = 30
time_overlap_fcd = 20
time_step = 3

time_seconds = num_points*time_step



data_total = {'RS': data_rs, 'COMP': data_comp}

data_empirical = {}


Ny_freq = 0.5*(num_points/time_seconds)
flp = 0.008
fhi = 0.08

filter_band = (flp, fhi)

num_win = 25

#Grad = GradientMaps(approach='dm', kernel='cosine')  
Grad = GradientMaps(approach='dm')  

#FCD_random = np.empty(shape = (2, num_subjects, num_win, num_win))


for key in data_total:
    
    ts_rs_aal = data_total[key]
    
    FCs = np.empty(shape = (num_subjects, num_nodes, num_nodes))
    
    FCDs = np.empty(shape = (num_subjects, num_win, num_win))
    
    
    #TEs = np.empty(shape = (num_subjects, 7, 90, 90))
    #Leida_vectors = np.empty(shape = (num_subjects, num_points, num_nodes))
    
    nodes_freq = np.zeros(shape = (num_nodes,))
    
    
    FCD_hist = []
    Synchronization = []
    Metastability = []

    #Cos_dist_mean = np.empty(shape = (num_subjects,  840, num_nodes, num_nodes))

    Meta_synch = np.zeros(shape=(num_subjects, 2))


    ts_rs_detrend = np.zeros(shape = (num_subjects, num_nodes, num_points))
    count = 0  
    for subject in range(num_subjects):
        
        ts_rs = ts_rs_aal[subject + 5]
        #ts_rs = ts_rs_aal[subject]
        
        for nodo in range(num_nodes):
            
            signal = sp.signal_detrend(ts_rs[nodo])
    
            #signal = sp.signal_detrend(ts_rs[subject][nodo])
            
            sos = butter(2, [filter_band[0]/Ny_freq, filter_band[1]/Ny_freq], 'bandpass', output='sos')
            filtered = sosfilt(sos, signal)
          
            f, Pxx_den = welch(filtered, fs= 1)
            maxValue = np.argmax(Pxx_den)
            freq = f[maxValue]
            nodes_freq[nodo] += freq            
            
            
            ts_rs_detrend[subject][nodo] = filtered
        
        FCs[subject] = np.corrcoef(ts_rs_detrend[subject])

        FCD = FCD_cal(np.transpose(ts_rs_detrend[subject]), time_window_duration = time_window_fcd, time_window_overlap = time_overlap_fcd , tstep_of_data = time_step )
        
        FCDs[subject] = FCD
        # if key == 'PCB':
        #     FCD_random[0, subject] = FCD
        # else:
        #     FCD_random[1, subject] = FCD
        

        
        #length_windows = int(60*4)
        #TEs[subject] = pair_granger_norm_6(ts_rs_detrend[subject], NLags, length_windows, 2)
        
        num_windows = len(FCD)
        FCD_hist.append(FCD[np.triu_indices(25, k = 1)])
    
        phase = phases(np.transpose(ts_rs_detrend[subject]))

        #leida = Leida(phase)
        kuramoto = Kuramoto(phase)
    
        Synchronization.append(np.mean(kuramoto))
        Metastability.append(np.std(kuramoto, ddof = 1))
        
        # Cos_dist = Cos_distance(phase)
        # Cos_dist_mean[subject] = Cos_dist
        
        # Leida_vectors[subject] = leida
        
        
        # transfer_entropy[subject] = transfer_entropy_norm(ts_rs_detrend[subject], NLags)
        
    #     Phases_mean[subject] = phase
    
    # transfer_entropy_mean = transfer_entropy.mean(axis = 0)
    # Cos_dist_mean = Cos_dist_mean.mean(axis = 0)
    
    # eigen_values, eigen_vectors = np.linalg.eig(Cos_dist_mean)
    # max_eigenval = np.argmax(eigen_values, axis = 1)
    

    # v = np.arange(len(max_eigenval))

    # max_eigenvector = eigen_vectors[v, :, max_eigenval]    
    
    # print('max eigen', max_eigenvector.shape)
    # print(Leida_vectors.shape)
    
    
    # Leida_vectors = max_eigenvector
    
    #Phases_mean_2 = np.average(np.exp(1j*Phases_mean), axis = 0)
    
    # Phases_mean_2 = np.arctan(np.imag(Phases_mean_2)/ np.real(Phases_mean_2))
    
    
    # Leida_vectors = np.real(Leida(Phases_mean_2))

    #Leida_vectors_mean = Leida_vectors.mean(axis = 0)
    

    fisher_FC = np.arctanh(FCs)
    fisher_mean_FC = np.mean(fisher_FC, axis = 0)
    FC_mean = np.tanh(fisher_mean_FC)
    
    fisher_FCD = np.arctanh(FCDs)
    fisher_mean_FCD = np.mean(fisher_FCD, axis = 0)
    FCD_mean = np.tanh(fisher_mean_FCD)
    
    nodes_freq = 2*np.pi*nodes_freq/num_subjects
    
    
    FCD_hist = np.array(list(chain.from_iterable(FCD_hist)))
    
    
    #FC_mean = stats.zscore(FC_mean)
    
    runs = 100
    gradients = np.empty((runs, num_nodes, 10))
    lambdas = np.empty((runs, num_nodes, 10))
    grad_aligned = np.empty((runs, num_nodes, 10))
    
    for run in range(runs):
        Gemp = Grad.fit(FC_mean)
        aligned = procrustes_alignment([reference10, Gemp.gradients_], n_iter = 30)
        
        gradients[run] =  Gemp.gradients_
        lambdas[run] = Gemp.lambdas_
        grad_aligned[run] = aligned[1]
        
    
    # gradients_mean = gradients.mean(axis = 0)
    # lambdas_mean = lambdas.mean(axis = 0)
    # grad_aligned_mean = grad_aligned.mean(axis = 0)
    
    
    #data_empirical[f'Leida_{key}'] = Leida_vectors
    
 
    
    data_empirical[f'freq_{key}'] = nodes_freq
    

    
    data_empirical[f'FC_{key}'] = FC_mean
    data_empirical[f'FCD_{key}'] = FCD_mean
    data_empirical[f'FCD_hist_{key}'] = FCD_hist

    
    data_empirical[f'Synchronization_{key}'] = np.mean(Synchronization)
    data_empirical[f'Synchronization_std_{key}'] = np.std(Synchronization, ddof = 1)
    
    data_empirical[f'Metastability_{key}'] = np.mean(Metastability)
    data_empirical[f'Metastability_std_{key}'] = np.std(Metastability, ddof = 1)
    
    data_empirical[f'gradients_{key}'] = gradients
    data_empirical[f'lambdas_{key}'] = lambdas
    data_empirical[f'gradientes_aligned_{key}'] = grad_aligned 
    
    
# plt.hist(gradientes_aligned_3[:, 0], alpha = 0.5)
# plt.hist(gradientes_aligned_2[:, 0], alpha = 0.5)
# plt.hist(reference10[:, 0], alpha = 0.5)
    
savemat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_new.mat", data_empirical)


#%%
"""Here is teh computation of the empirical observables concatenating the time series"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.io import loadmat, savemat
import signal_process as sp
from Functional_Connectivity import *
from Kuramoto import *
from itertools import chain
from scipy.signal import butter, sosfilt, welch
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.gradient.alignment import procrustes_alignment


data =  loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")



# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
reference10 = reference[:,0:10]


data_rs_raw = data['TS_RS'][0]
data_comp_raw = data['TS_COMP'][0]

data_total_raw = {'RS': data_rs_raw, 'COMP': data_comp_raw}

data_rs = np.empty((12, 1000, 1023))
data_comp = np.empty((12, 1000, 1023))


num_points = 93
num_subjects = 100
num_nodes = 1000

for key in data_total_raw:
    
    ts_rs_aal = data_total_raw[key]
    count_subset = 0
    for subset in range(0, 132, 11):
        count_subject = 0
        for subject in range(num_subjects):
            
            ts_rs = ts_rs_aal[subset + subject + 5]
            if key == 'RS':
                print(count_subset, count_subject*num_points, (count_subject +  1)*num_points)
                
                #data_rs[count_subset, :, count_subject*num_points:(count_subject +  1)*num_points] = ts_rs
            
            else: pass
                #data_comp[count_subset, :, count_subject*num_points:(count_subject +  1)*num_points] = ts_rs
               
            
            count_subject += 1
        
        count_subset += 1

num_points = 93
num_subjects = 100
num_nodes = 1000

time_window_fcd  = 30
time_overlap_fcd = 20
time_step = 3

time_seconds = num_points*time_step



data_total = {'RS': data_rs, 'COMP': data_comp}

data_empirical = {}


Ny_freq = 0.5*(num_points/time_seconds)
flp = 0.008
fhi = 0.08

filter_band = (flp, fhi)

num_win = 25

#Grad = GradientMaps(approach='dm', kernel='cosine')  
Grad = GradientMaps(approach='dm')  

#FCD_random = np.empty(shape = (2, num_subjects, num_win, num_win))


for key in data_total:
    
    ts_rs_aal = data_total[key]
    
    FCs = np.empty(shape = (num_subjects, num_nodes, num_nodes))
    
    FCDs = np.empty(shape = (num_subjects, num_win, num_win))
    
    
    #TEs = np.empty(shape = (num_subjects, 7, 90, 90))
    #Leida_vectors = np.empty(shape = (num_subjects, num_points, num_nodes))
    
    nodes_freq = np.zeros(shape = (num_nodes,))
    
    
    FCD_hist = []
    Synchronization = []
    Metastability = []

    #Cos_dist_mean = np.empty(shape = (num_subjects,  840, num_nodes, num_nodes))

    Meta_synch = np.zeros(shape=(num_subjects, 2))


    ts_rs_detrend = np.zeros(shape = (num_subjects, num_nodes, num_points))
    count = 0  
    for subject in range(num_subjects):
        
        ts_rs = ts_rs_aal[subject + 5]
        #ts_rs = ts_rs_aal[subject]
        
        for nodo in range(num_nodes):
            
            signal = sp.signal_detrend(ts_rs[nodo])
    
            #signal = sp.signal_detrend(ts_rs[subject][nodo])
            
            sos = butter(2, [filter_band[0]/Ny_freq, filter_band[1]/Ny_freq], 'bandpass', output='sos')
            filtered = sosfilt(sos, signal)
          
            f, Pxx_den = welch(filtered, fs= 1)
            maxValue = np.argmax(Pxx_den)
            freq = f[maxValue]
            nodes_freq[nodo] += freq            
            
            
            ts_rs_detrend[subject][nodo] = filtered
        
        FCs[subject] = np.corrcoef(ts_rs_detrend[subject])

        FCD = FCD_cal(np.transpose(ts_rs_detrend[subject]), time_window_duration = time_window_fcd, time_window_overlap = time_overlap_fcd , tstep_of_data = time_step )
        
        FCDs[subject] = FCD
        # if key == 'PCB':
        #     FCD_random[0, subject] = FCD
        # else:
        #     FCD_random[1, subject] = FCD
        

        
        #length_windows = int(60*4)
        #TEs[subject] = pair_granger_norm_6(ts_rs_detrend[subject], NLags, length_windows, 2)
        
        num_windows = len(FCD)
        FCD_hist.append(FCD[np.triu_indices(25, k = 1)])
    
        phase = phases(np.transpose(ts_rs_detrend[subject]))

        #leida = Leida(phase)
        kuramoto = Kuramoto(phase)
    
        Synchronization.append(np.mean(kuramoto))
        Metastability.append(np.std(kuramoto, ddof = 1))
        
        # Cos_dist = Cos_distance(phase)
        # Cos_dist_mean[subject] = Cos_dist
        
        # Leida_vectors[subject] = leida
        
        
        # transfer_entropy[subject] = transfer_entropy_norm(ts_rs_detrend[subject], NLags)
        
    #     Phases_mean[subject] = phase
    
    # transfer_entropy_mean = transfer_entropy.mean(axis = 0)
    # Cos_dist_mean = Cos_dist_mean.mean(axis = 0)
    
    # eigen_values, eigen_vectors = np.linalg.eig(Cos_dist_mean)
    # max_eigenval = np.argmax(eigen_values, axis = 1)
    

    # v = np.arange(len(max_eigenval))

    # max_eigenvector = eigen_vectors[v, :, max_eigenval]    
    
    # print('max eigen', max_eigenvector.shape)
    # print(Leida_vectors.shape)
    
    
    # Leida_vectors = max_eigenvector
    
    #Phases_mean_2 = np.average(np.exp(1j*Phases_mean), axis = 0)
    
    # Phases_mean_2 = np.arctan(np.imag(Phases_mean_2)/ np.real(Phases_mean_2))
    
    
    # Leida_vectors = np.real(Leida(Phases_mean_2))

    #Leida_vectors_mean = Leida_vectors.mean(axis = 0)
    

    fisher_FC = np.arctanh(FCs)
    fisher_mean_FC = np.mean(fisher_FC, axis = 0)
    FC_mean = np.tanh(fisher_mean_FC)
    
    fisher_FCD = np.arctanh(FCDs)
    fisher_mean_FCD = np.mean(fisher_FCD, axis = 0)
    FCD_mean = np.tanh(fisher_mean_FCD)
    
    nodes_freq = 2*np.pi*nodes_freq/num_subjects
    
    
    FCD_hist = np.array(list(chain.from_iterable(FCD_hist)))
    
    runs = 100
    gradients = np.empty((runs, num_nodes, 10))
    lambdas = np.empty((runs, num_nodes, 10))
    grad_aligned = np.empty((runs, num_nodes, 10))
    
    for run in range(runs):
        Gemp = Grad.fit(FC_mean)
        aligned = procrustes_alignment([reference10, Gemp.gradients_, n_iter = 30])
        
        gradients[run] =  Gemp.gradients_
        lambdas[run] = Gemp.lambdas_
        grad_aligned[run] = aligned[1]
        
    
    # gradients_mean = gradients.mean(axis = 0)
    # lambdas_mean = lambdas.mean(axis = 0)
    # grad_aligned_mean = grad_aligned.mean(axis = 0)
    
    
    #data_empirical[f'Leida_{key}'] = Leida_vectors
    
 
    
    data_empirical[f'freq_{key}'] = nodes_freq
    

    
    data_empirical[f'FC_{key}'] = FC_mean
    data_empirical[f'FCD_{key}'] = FCD_mean
    data_empirical[f'FCD_hist_{key}'] = FCD_hist

    
    data_empirical[f'Synchronization_{key}'] = np.mean(Synchronization)
    data_empirical[f'Synchronization_std_{key}'] = np.std(Synchronization, ddof = 1)
    
    data_empirical[f'Metastability_{key}'] = np.mean(Metastability)
    data_empirical[f'Metastability_std_{key}'] = np.std(Metastability, ddof = 1)
    
    data_empirical[f'gradients_{key}'] = gradients
    data_empirical[f'lambdas_{key}'] = lambdas
    data_empirical[f'gradientes_aligned_{key}'] = grad_aligned 
    
    
savemat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_concat.mat", data_empirical)

