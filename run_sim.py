from scipy.io import loadmat
import numpy as np
from Simulations_yoni import *
from metrics import * 
from scipy.stats import gamma
import pandas as pd
import matplotlib.pyplot as plt
from brainspace.datasets import load_conte69
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.gradient.alignment import procrustes_alignment
from scipy.io import loadmat
from scipy.spatial import distance
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres



#load all needed data for the Simulation 
data_SC = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")
data_freq = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\hopf_freq_SCH1000_RS")
data_empirical = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_zscore")

labeling = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\schaefer_1000_conte69.csv", delimiter=',')

condition = 'RS'

FCD_empirical = data_empirical.get(f'FCD_{condition}', None)    
FC_empirical = data_empirical.get(f'FC_{condition}', None)    
FCt_empirical = data_empirical.get(f'FCt_{condition}', None)

FCD_hist_empirical = (data_empirical.get(f'FCD_hist_{condition}', None)).squeeze()
Synchronization_empirical = data_empirical.get(f'Synchronization_{condition}', None)[0]
Metastability_empirical = data_empirical.get(f'Metastability_{condition}', None)[0]

gradients_empirical_total = data_empirical.get(f'gradients_{condition}', None)
aligments_empirical_total = data_empirical.get(f'gradientes_aligned_{condition}', None)
lambdas_empirical_total = data_empirical.get(f'lambdas_{condition}', None)

aligments_empirical = aligments_empirical_total.mean(axis = 0)
gradients_empirical = gradients_empirical_total.mean(axis = 0)
lambdas_empirical = lambdas_empirical_total.mean(axis = 0)


# and load the conte69 surfaces
surf_lh, surf_rh = load_conte69()


grad = map_to_labels(aligments_empirical[:, 0], labeling, mask=labeling != 0, fill=np.nan)

# # Plot first gradient on the cortical surface.
# plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(800, 200))
plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r', color_bar=True, label_text=['Grad1_emp'], zoom=1.55)


# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
reference10 = reference[:,0:10]

SC = data_SC.get('SC', None)    #structure connectivity matrix

#q = np.quantile(SC[np.triu_indices(1000, k = 1)], q = 0.99)
#SC = (SC/q)*0.2

SC = (SC/SC.max())*0.2


numNodes = len(SC)  #number of nodes
freq = data_empirical.get(f'freq_{condition}', None)  #frequencies of awake state
K = np.sum(SC, axis = 0) # constant to use in the models (coupling term)

tStep = 0.1


SC = SC + 1j*np.zeros_like(SC)

SC = np.array(SC, dtype = 'F')


#attributes of the Simulation Class (Hopf in this case)

flp = 0.008
fhi = 0.08

t_final = 479
thermalization_time = 200
sim = Hopf_simulation(t1= 0, t2=t_final, t_thermal = thermalization_time, rescale_time=3, t_step = tStep, num_nodes = numNodes, integration_method = 'euler_maruyama', filter_band = (flp, fhi))
sim.FCD_parameters(30, 20, 3)


time = np.arange(0, t_final, tStep)


a = 0.01
G = 0.75

Grad = GradientMaps(approach='dm', alignment='procrustes')


subjects = 20



fcd = np.zeros(shape = (subjects, 300))

FCs = np.zeros(shape = (subjects, numNodes, numNodes))
FCDs = np.zeros(shape = (subjects, len(FCD_empirical), len(FCD_empirical)))

metas = []
synch = []
for sub in range(subjects):
    print('subject:', sub)
    
    ic =  np.random.uniform(-0.1, 0.1, size = (1, numNodes)) + 1j*np.random.uniform(-0.1, 0.1, size = (1, numNodes))

    ic = np.array(ic, dtype = 'F')
    
    sim.model_parameters(a = a,  M = SC, frequencies = freq, constant = K , G = G, noise_amp = 0.04*np.sqrt(tStep))


    sim.initial_conditions(ic)
    sim.run_sim() 

    fcd[sub] = sim.FCD[np.triu_indices(25, k = 1)]
    FCs[sub] = sim.FC
    FCDs[sub] = sim.FCD
    #FC_temp[sub] =   FC_temp_cal(sim.t_series, time_window_duration = 240, time_window_overlap = 0 , tstep_of_data = 2 )
    
    metas.append(sim.Metastability)
    synch.append(sim.Synchronization)
    


fisher_FC = np.arctanh(FCs)
fisher_FC_mean = np.mean(fisher_FC, axis = 0)

FC_mean = np.tanh(fisher_FC_mean)

fisher_FCD = np.arctanh(FCDs)
fisher_FCD_mean = np.mean(fisher_FCD, axis = 0)

FCD_mean = np.tanh(fisher_FCD_mean)



runs = 20
gradients = np.empty((runs, numNodes, 10))
lambdas = np.empty((runs, numNodes, 10))
grad_aligned_daniel = np.empty((runs, numNodes, 10))
grad_aligned_emp = np.empty((runs, numNodes, 10))

for run in range(runs):
    Gsim = Grad.fit(FC_mean)
    aligned_daniel = procrustes_alignment([reference10, Gsim.gradients_])
    aligned_emp = procrustes_alignment([gradients_empirical, Gsim.gradients_])                
    
    
    gradients[run] =  Gsim.gradients_
    lambdas[run] = Gsim.lambdas_
    grad_aligned_daniel[run] = aligned_daniel[1]
    grad_aligned_emp[run] = aligned_emp[1]
    
gradients_sim = gradients.mean(axis = 0)
lambdas_sim = lambdas.mean(axis = 0)
grad_aligned_sim_daniel = grad_aligned_daniel.mean(axis = 0)
grad_aligned_sim_emp = grad_aligned_emp.mean(axis = 0)


SSIM_FC = 1 - ssim(FC_mean, FC_empirical, data_range=2)
SSIM_FCD = 1 - ssim(FCD_mean, FCD_empirical,  data_range=2)
frob_FCD = np.linalg.norm(FCD_mean - FCD_empirical)/np.linalg.norm(FCD_empirical)
frob_FC= np.linalg.norm(FC_mean - FC_empirical)/np.linalg.norm(FC_empirical)

KS= stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
Synch_metric =  np.abs((np.mean(synch) - Synchronization_empirical))/Synchronization_empirical
Metas_metric= np.abs((np.mean(metas) - Metastability_empirical))/Metastability_empirical


SSIM_grad = 1 - ssim(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0], data_range=2)

Frob_grad = np.linalg.norm(grad_aligned_sim_daniel[:, 0] - aligments_empirical[:, 0])/np.linalg.norm(aligments_empirical[:, 0])

Corr_grad = np.corrcoef(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])[0, 1]

Cos_grad = distance.cosine(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])

grad = map_to_labels(grad_aligned_sim_daniel[:, 0], labeling, mask=labeling != 0, fill=np.nan)

# Plot first gradient on the cortical surface.
#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(800, 200))

#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r', color_bar=True, label_text=['Grad1'], zoom=1.55)


# plt.imshow(FC_empirical)
# plt.colorbar()
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\Simulation_2\FC_emp.png")

from mpl_toolkits.mplot3d import Axes3D
from brainspace.plotting import plot_hemispheres

def gradient_in_euclidean(gradients, surface=None, parcellation=None):
    if gradients.shape[1] != 3 and gradients.shape[1] != 2:
        raise ValueError('Input matrix must be numeric with two or three columns.')

    h = {}
    h['figure'] = plt.figure(figsize=(9, 9), facecolor='white')
    h['axes_scatter'] = plt.subplot(221, position=[0.38, 0.6, 0.3, 0.3])

    if gradients.shape[1] == 2:
        cart = (gradients - np.mean(gradients, axis=0)) / np.max(np.abs(gradients - np.mean(gradients)))
        th, r = np.arctan2(cart[:, 1], cart[:, 0]), np.linalg.norm(cart, axis=1)
        r = r / 2 + 0.5
        C = np.vstack((np.cos(0.75 * (th + 0 * np.pi)),
                       np.cos(0.75 * th - 0.5 * np.pi),
                       np.cos(0.75 * th + 0.5 * np.pi))).T * r[:, np.newaxis]
        C = C / np.max(C)
        C[C < 0] = 0
        h['scatter'] = plt.scatter(gradients[:, 0], gradients[:, 1], c=C, marker='.')
    else:
        C = (gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients))
        h['scatter'] = plt.scatter(gradients[:, 0], gradients[:, 1], gradients[:, 2], c=C, marker='.')
        plt.ylabel('Gradient 3')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Gradient 1')
    plt.ylabel('Gradient 2')

    if surface is not None:
        if parcellation is not None:
            D_full = parcellation
        D_full[np.all(np.isnan(D_full), axis=1)] = 0.7
        S = surface
        if not isinstance(S, list):
            S = [S]
        if len(S) > 3:
            raise ValueError('More than two surfaces are not accepted.')
        N1 = sum([s['coord'].shape[1] for s in S])
        N2 = D_full.shape[0]
        if N1 != N2:
            raise ValueError('Number of vertices on the surface and number of data points do not match.')
        D = [D_full[:S[0]['coord'].shape[1]]]
        if len(S) == 2:
            D.append(D_full[S[0]['coord'].shape[1]:])
        for ii, s in enumerate(S):
            ax = plt.subplot(222 + ii)
            h['trisurf'] = ax.plot_trisurf(s['tri'], s['coord'][0], s['coord'][1], s['coord'][2], D[ii],
                                           edgecolor='none', cmap=plt.cm.gray)
            plt.setp(ax, visible=False, aspect='equal', clim=[0, D_full.shape[1]])
            ax.view_init(-90 if ii == 0 else 90, 0)
            plt.colorbar(h['trisurf'])
            plt.tight_layout(pad=2)
            plt.subplots_adjust(wspace=0.5)

            for cam_ax in range(len(S)):
                ax = plt.subplot(222 + cam_ax)
                h['camlight'] = ax.light = plt.light_sources[cam_ax]()

    return h

gradient_in_euclidean(aligments_empirical[:, 0:2])
plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\Simulation_2\grad_euclidean_emp.png")


# plt.hist(SC[np.triu_indices(90, k = 1)])
# plt.xscale('log')