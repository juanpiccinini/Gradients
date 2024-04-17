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
from scipy.io import loadmat
from scipy.spatial import distance
from brainspace.gradient.alignment import procrustes_alignment
from scipy import stats




#load all needed data for the Simulation 
data_SC = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")
data_empirical = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_26")

# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
#reference10 = reference[:,0:10]
reference10 = reference



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

lambdas_empirical_3 = lambdas_empirical[0, 0:3]/lambdas_empirical[0, 0:3].sum()

grad_emp_weight  =  gradients_empirical[:, 0:3]@(lambdas_empirical_3*np.identity(3))

SC = data_SC.get('SC', None)    #structure connectivity matrix

#q = np.quantile(SC[np.triu_indices(1000, k = 1)], q = 0.99)
#SC = (SC/q)*0.2

SC = (SC/SC.max())*0.2


numNodes = len(SC)  #number of nodes
freq = data_empirical.get(f'freq_{condition}', None)[0]  #frequencies of awake state
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


num_experiments = 10
matrices_mean_FCD = np.empty((num_experiments, len(FCD_empirical), len(FCD_empirical)))


a_all = np.arange(-0.1, 0.1, 0.01)
a_all = [-0.02]
G_all = np.arange(0, 3, 0.15)
#G_all = [0.75]


matrices_mean_FCD = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FC = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FC_frob = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FCD_frob = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_FC_corr = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_FCD_KS = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_Synch = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_Metas = np.empty((num_experiments, len(a_all), len(G_all)))


matrices_mean_gradient_alig_daniel_SSIM_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Frob_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Corr_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Cosine_1 = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_gradient_alig_emp_SSIM_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Frob_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Corr_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Cosine_1 = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_gradient_alig_daniel_SSIM_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Frob_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Corr_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Cosine_3 = np.empty((num_experiments, len(a_all), len(G_all)))

n_components = 26

Grad = GradientMaps(n_components = n_components, approach='dm')


#FC_empirical = stats.zscore(FC_empirical)

subjects = 15

for exper in range(num_experiments):
    SSIM_FC = np.zeros((len(a_all), len(G_all)))
    SSIM_FCD = np.zeros((len(a_all), len(G_all)))
    frob_FC = np.zeros((len(a_all), len(G_all)))
    frob_FCD = np.zeros((len(a_all), len(G_all)))
    corr_FC = np.zeros((len(a_all), len(G_all)))
    
    KS = np.zeros((len(a_all), len(G_all)))
    Synch_metric = np.zeros((len(a_all), len(G_all)))
    Metas_metric = np.zeros((len(a_all), len(G_all)))
    
    SSIM_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    
    SSIM_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    

    SSIM_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    

    count_raw = 0
    count_col = 0
    
    for a in a_all:
        
        for G in G_all:



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
                
                # SSIM_FCD_dict['SSIM'].append(1 - ssim(sim.FCD, FCD_empirical))
                # SSIM_FCD_dict['alpha'].append(alpha)
                # SSIM_FCD_dict['beta'].append(beta)
                # metric = Metrics(sim)
    
    
            # fisher_FC_temp = np.arctanh(FC_temp)
            # fisher_mean_FC_temp = np.mean(fisher_FC_temp, axis = 0)
            # FC_temp_mean = np.tanh(fisher_mean_FC_temp)
                
            
            fisher_FC = np.arctanh(FCs)
            fisher_FC_mean = np.mean(fisher_FC, axis = 0)
            
            FC_mean = np.tanh(fisher_FC_mean)
            
            #FC_mean = stats.zscore(FC_mean)
            
            fisher_FCD = np.arctanh(FCDs)
            fisher_FCD_mean = np.mean(fisher_FCD, axis = 0)
            
            FCD_mean = np.tanh(fisher_FCD_mean)

            runs = 5
            gradients = np.empty((runs, numNodes, n_components))
            lambdas = np.empty((runs, numNodes, n_components))
            grad_aligned_daniel = np.empty((runs, numNodes, n_components))
            grad_aligned_emp = np.empty((runs, numNodes, n_components))
            
            for run in range(runs):
                Gsim = Grad.fit(FC_mean)
                aligned_daniel = procrustes_alignment([reference10, Gsim.gradients_], n_iter = 30)
                aligned_emp = procrustes_alignment([gradients_empirical, Gsim.gradients_], n_iter = 30)                
                
                
                gradients[run] =  Gsim.gradients_
                lambdas[run] = Gsim.lambdas_
                grad_aligned_daniel[run] = aligned_daniel[1]
                grad_aligned_emp[run] = aligned_emp[1]
                
            gradients_sim = gradients.mean(axis = 0)
            lambdas_sim = lambdas.mean(axis = 0)
            grad_aligned_sim_daniel = grad_aligned_daniel.mean(axis = 0)
            grad_aligned_sim_emp = grad_aligned_emp.mean(axis = 0)
            #grad_aligned_sim = gradients_sim


            grad_aligned_sim_daniel /= np.linalg.norm(grad_aligned_sim_daniel, axis = 0)
            grad_aligned_sim_emp /= np.linalg.norm(grad_aligned_sim_emp, axis = 0)
            
            
            print(np.linalg.norm(grad_aligned_sim_emp, axis = 0))
            #fcd_total[count_raw, count_col] = fcd.flatten()
            
            # SSIM_FCt_temporal = []
            # frob_FCt_temporal = []
            # for i in range(2, 7):
            #     SSIM_FCt_temporal.append(1 - ssim(FC_temp_mean[i], FCt_empirical[i]))
            #     frob_FCt_temporal.append(np.linalg.norm(FC_temp_mean[i] - FCt_empirical[i])/np.linalg.norm(FCt_empirical[i]))
            
            # SSIM_FCt[count_raw, count_col] = np.mean(SSIM_FCt_temporal)
            # frob_FCt[count_raw, count_col] = np.mean(frob_FCt_temporal)

            
            SSIM_FC[count_raw, count_col] = 1 - ssim(FC_mean, FC_empirical, data_range=2)
            SSIM_FCD[count_raw, count_col] = 1 - ssim(FCD_mean, FCD_empirical,  data_range=2)
            frob_FCD[count_raw, count_col] = np.linalg.norm(FCD_mean - FCD_empirical)/np.linalg.norm(FCD_empirical)
            frob_FC[count_raw, count_col] = np.linalg.norm(FC_mean - FC_empirical)/np.linalg.norm(FC_empirical)
            corr_FC[count_raw, count_col] = (1 - np.corrcoef(np.tanh(FC_empirical[np.triu_indices(numNodes, k = 1)]), np.tanh(FC_mean[np.triu_indices(numNodes, k = 1)]))[0, 1])

            
            KS[count_raw, count_col] = stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
            Synch_metric[count_raw, count_col] =  np.abs((np.mean(synch) - Synchronization_empirical))/Synchronization_empirical
            Metas_metric[count_raw, count_col] = np.abs((np.mean(metas) - Metastability_empirical))/Metastability_empirical
            
            
            SSIM_grad_alig_daniel_1[count_raw, count_col] = 1 - ssim(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0], data_range=2)
        
            Frob_grad_alig_daniel_1[count_raw, count_col] = np.linalg.norm(grad_aligned_sim_daniel[:, 0] - aligments_empirical[:, 0])/np.linalg.norm(aligments_empirical[:, 0])
        
            Corr_grad_alig_daniel_1[count_raw, count_col] = np.corrcoef(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])[0, 1]
    
            Cos_grad_alig_daniel_1[count_raw, count_col] = distance.cosine(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])
            


            SSIM_grad_alig_emp_1[count_raw, count_col] = 1 - ssim(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0], data_range=2)
        
            Frob_grad_alig_emp_1[count_raw, count_col] = np.linalg.norm(grad_aligned_sim_emp[:, 0] - gradients_empirical[:, 0])/np.linalg.norm(gradients_empirical[:, 0])
        
            Corr_grad_alig_emp_1[count_raw, count_col] = np.corrcoef(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0])[0, 1]
    
            Cos_grad_alig_emp_1[count_raw, count_col] = distance.cosine(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0])
            

        
        
            SSIM_grad_alig_daniel_3[count_raw, count_col] = 1 - (ssim(grad_aligned_sim_daniel, aligments_empirical, data_range=2))
        
        
        
        
            # SSIM_grad_alig_daniel_3[count_raw, count_col] = 1 - ( lambdas_empirical_3[0]*ssim(grad_aligned_sim[:, 0], gradients_empirical[:, 0], data_range=2) + \
            #                                              lambdas_empirical_3[1]*ssim(grad_aligned_sim[:, 1], gradients_empirical[:, 1], data_range=2) + \
            #                                                  lambdas_empirical_3[2]*ssim(grad_aligned_sim[:, 2], gradients_empirical[:, 2], data_range=2) )
        
            
            # grad_sim_weight  =  grad_aligned_sim[:, 0:3]@(lambdas_empirical_3*np.identity(3))
        
            # Frob_grad_alig_daniel_3[count_raw, count_col] = np.linalg.norm(grad_sim_weight - grad_emp_weight)/np.linalg.norm(grad_emp_weight)
        
            # Corr_grad_alig_daniel_3[count_raw, count_col] = lambdas_empirical_3[0]*(np.corrcoef(grad_aligned_sim[:, 0], gradients_empirical[:, 0])[0, 1]) + \
            #                                        lambdas_empirical_3[1]*(np.corrcoef(grad_aligned_sim[:, 1], gradients_empirical[:, 1])[0, 1]) + \
            #                                         lambdas_empirical_3[2]*(np.corrcoef(grad_aligned_sim[:, 2], gradients_empirical[:, 2])[0, 1])
            
            # Cos_grad_alig_daniel_3[count_raw, count_col] = lambdas_empirical_3[0]*distance.cosine(grad_aligned_sim[:, 0], gradients_empirical[:, 0]) +\
            #                                         lambdas_empirical_3[1]*distance.cosine(grad_aligned_sim[:, 1], gradients_empirical[:, 1]) +\
            #                                             lambdas_empirical_3[2]*distance.cosine(grad_aligned_sim[:, 2], gradients_empirical[:, 2])
          
                    
            #KS_post[count_raw, count_col] = stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
   
            
            print(a, G, SSIM_grad_alig_daniel_1[count_raw, count_col],  Frob_grad_alig_daniel_1[count_raw, count_col], Corr_grad_alig_daniel_1[count_raw, count_col])
    
            
            if count_col != len(G_all) -1:
                count_col += 1
            else:
                count_col = 0
        count_raw += 1
    matrices_mean_FCD[exper] = SSIM_FCD
    matrices_mean_FC[exper] = SSIM_FC
    matrices_mean_FCD_frob[exper] = frob_FCD
    matrices_mean_FC_frob[exper] = frob_FC   
    matrices_mean_FC_corr[exper] = corr_FC 
 
    
    matrices_mean_FCD_KS[exper]  = KS
    matrices_mean_Synch[exper]  = Synch_metric

    matrices_mean_Metas[exper]  = Metas_metric
    
    matrices_mean_gradient_alig_daniel_SSIM_1[exper]= SSIM_grad_alig_daniel_1

    matrices_mean_gradient_alig_daniel_Frob_1[exper] = Frob_grad_alig_daniel_1
    
    matrices_mean_gradient_alig_daniel_Corr_1[exper] = Corr_grad_alig_daniel_1

    matrices_mean_gradient_alig_daniel_Cosine_1[exper] = Cos_grad_alig_daniel_1
    
    
    matrices_mean_gradient_alig_emp_SSIM_1[exper]= SSIM_grad_alig_emp_1

    matrices_mean_gradient_alig_emp_Frob_1[exper] = Frob_grad_alig_emp_1
    
    matrices_mean_gradient_alig_emp_Corr_1[exper] = Corr_grad_alig_emp_1

    matrices_mean_gradient_alig_emp_Cosine_1[exper] = Cos_grad_alig_emp_1
    
    matrices_mean_gradient_alig_daniel_SSIM_3[exper]= SSIM_grad_alig_daniel_3

    # matrices_mean_gradient_alig_daniel_Frob_3[exper] = Frob_grad_3
    
    # matrices_mean_gradient_alig_daniel_Corr_3[exper] = Corr_grad_3

    # matrices_mean_gradient_alig_daniel_Cosine_3[exper] = Cos_grad_3



# plt.imshow(matrices_mean_FC_corr[0])
# plt.colorbar()

# num_simulation = 4
# path = f"D:\Juan\Facultad\Doctorado\Gradients\Simulation_{num_simulation}"
 
# # checking if the directory demo_folder  
# # exist or not. 
# if not os.path.exists(path): 
      
#     # if the demo_folder directory is not present  
#     # then create it. 
#     os.makedirs(path)

# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FCD_{condition}.npy", matrices_mean_FCD)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FCD_{condition}.npy", matrices_mean_FCD_frob)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FC_{condition}.npy", matrices_mean_FC)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FC_{condition}.npy", matrices_mean_FC)

# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\KS_FCD_{condition}.npy", matrices_mean_FCD_KS)


# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_SSIM_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Frob_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Corr_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Cosine_1)


# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_SSIM_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Frob_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Corr_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Cosine_1)




# plt.plot(G_all, matrices_mean_gradient_SSIM_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\SSIM_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Frob_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Frob_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Corr_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Corr_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Cosine_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Cosdist_1grad_aconst.png")
# plt.close()


#%% 

"""This cell is to simulate the grid of parameters using from 1 up to 26 
gradients saving them in a dictionary"""


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
from scipy.io import loadmat
from scipy.spatial import distance
from brainspace.gradient.alignment import procrustes_alignment
from scipy import stats




#load all needed data for the Simulation 
data_SC = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")
data_empirical = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_26")

# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
#reference10 = reference[:,0:10]
reference10 = reference



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
lambdas_empirical = lambdas_empirical_total.mean(axis = 0)[0, :]


SC = data_SC.get('SC', None)    #structure connectivity matrix

#q = np.quantile(SC[np.triu_indices(1000, k = 1)], q = 0.99)
#SC = (SC/q)*0.2

SC = (SC/SC.max())*0.2


numNodes = len(SC)  #number of nodes
freq = data_empirical.get(f'freq_{condition}', None)[0]  #frequencies of awake state
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


num_experiments = 2


a_all = np.arange(-0.1, 0.1, 0.01)
a_all = [-0.02]
G_all = np.arange(0, 3, 0.15)
#G_all = [0.75]


# a_all = [-0.1, 0, 0.1]

# G_all = [0, 1, 2]


n_components = 26

Grad = GradientMaps(n_components = n_components, approach='dm')


#FC_empirical = stats.zscore(FC_empirical)

subjects = 15


dict_simulations = dict()


    
dict_simulations = dict()
dict_simulations['SSIM_FC'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['SSIM_FCD'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['frob_FC'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['frob_FCD'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['corr_FC'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['KS'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['Synch_metric'] = np.zeros((num_experiments, len(a_all), len(G_all)))
dict_simulations['Metas_metric'] = np.zeros((num_experiments, len(a_all), len(G_all)))

dict_simulations['SSIM_grad_alig_daniel'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))
dict_simulations['Frob_grad_alig_daniel'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))
dict_simulations['Corr_grad_alig_daniel'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))

dict_simulations['SSIM_grad_alig_emp'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))
dict_simulations['Frob_grad_alig_emp'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))
dict_simulations['Corr_grad_alig_emp'] = np.zeros((num_experiments, n_components, len(a_all), len(G_all)))
dict_simulations['lambdas_empirical_weights']= []


for num_grad in range(n_components):
    lambdas_empirical_weights = lambdas_empirical[0:num_grad + 1]/lambdas_empirical[0:num_grad + 1].sum()
    dict_simulations['lambdas_empirical_weights'].append(lambdas_empirical_weights)

for exper in range(num_experiments):
    
    

    count_raw = 0
    count_col = 0
    
    for a in a_all:
        
        for G in G_all:



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
                
                metas.append(sim.Metastability)
                synch.append(sim.Synchronization)
                

                
            
            fisher_FC = np.arctanh(FCs)
            fisher_FC_mean = np.mean(fisher_FC, axis = 0)
            
            FC_mean = np.tanh(fisher_FC_mean)
            
            #FC_mean = stats.zscore(FC_mean)
            
            fisher_FCD = np.arctanh(FCDs)
            fisher_FCD_mean = np.mean(fisher_FCD, axis = 0)
            
            FCD_mean = np.tanh(fisher_FCD_mean)

            runs = 5
            gradients = np.empty((runs, numNodes, n_components))
            lambdas = np.empty((runs, numNodes, n_components))
            grad_aligned_daniel = np.empty((runs, numNodes, n_components))
            grad_aligned_emp = np.empty((runs, numNodes, n_components))
            
            for run in range(runs):
                Gsim = Grad.fit(FC_mean)
                aligned_daniel = procrustes_alignment([reference10, Gsim.gradients_], n_iter = 30)
                aligned_emp = procrustes_alignment([gradients_empirical, Gsim.gradients_], n_iter = 30)                
                
                
                gradients[run] =  Gsim.gradients_
                lambdas[run] = Gsim.lambdas_
                grad_aligned_daniel[run] = aligned_daniel[1]
                grad_aligned_emp[run] = aligned_emp[1]
                
            gradients_sim = gradients.mean(axis = 0)
            lambdas_sim = lambdas.mean(axis = 0)
            grad_aligned_sim_daniel = grad_aligned_daniel.mean(axis = 0)
            grad_aligned_sim_emp = grad_aligned_emp.mean(axis = 0)
            #grad_aligned_sim = gradients_sim


            grad_aligned_sim_daniel /= np.linalg.norm(grad_aligned_sim_daniel, axis = 0)
            grad_aligned_sim_emp /= np.linalg.norm(grad_aligned_sim_emp, axis = 0)
            
                        
            dict_simulations['SSIM_FC'][exper][count_raw, count_col] = 1 - ssim(FC_mean, FC_empirical, data_range=2)
            dict_simulations['SSIM_FCD'][exper][count_raw, count_col] = 1 - ssim(FCD_mean, FCD_empirical,  data_range=2)
            dict_simulations['frob_FC'][exper][count_raw, count_col] = np.linalg.norm(FC_mean - FC_empirical)/np.linalg.norm(FC_empirical)
            dict_simulations['corr_FC'][exper][count_raw, count_col] = (1 - np.corrcoef(FC_empirical[np.triu_indices(numNodes, k = 1)], FC_mean[np.triu_indices(numNodes, k = 1)])[0, 1])
            dict_simulations['KS'][exper][count_raw, count_col] = stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
            dict_simulations['Synch_metric'][exper][count_raw, count_col] = np.abs((np.mean(synch) - Synchronization_empirical))/Synchronization_empirical
            dict_simulations['Metas_metric'][exper][count_raw, count_col] = np.abs((np.mean(metas) - Metastability_empirical))/Metastability_empirical
            
            
            
            for num_grad in range(26):
            
                
                if num_grad == 0:
                    u_daniel = grad_aligned_sim_daniel[:, num_grad]
                    v_daniel = aligments_empirical[:, num_grad]
                    dict_simulations['SSIM_grad_alig_daniel'][exper, num_grad][count_raw, count_col] = 1 - ssim(u_daniel, v_daniel, data_range=2)
                    dict_simulations['Frob_grad_alig_daniel'][exper, num_grad][count_raw, count_col]  = np.mean((u_daniel - v_daniel)**2)
                    dict_simulations['Corr_grad_alig_daniel'][exper, num_grad][count_raw, count_col]  =  1 + np.corrcoef(u_daniel, v_daniel)[0, 1]
                    
         
                    u_emp = grad_aligned_sim_emp[:, num_grad ]
                    v_emp = gradients_empirical[:, num_grad]
                    
                    dict_simulations['SSIM_grad_alig_emp'][exper, num_grad][count_raw, count_col]  = 1 - ssim(u_emp, v_emp, data_range=2)
                    dict_simulations['Frob_grad_alig_emp'][exper, num_grad][count_raw, count_col] =  np.mean((u_emp - v_emp)**2)
                    dict_simulations['Corr_grad_alig_emp'][exper, num_grad][count_raw, count_col]  =  1 + np.corrcoef(u_emp, v_emp)[0,1]
    
                

                
                else:
                    weights = dict_simulations['lambdas_empirical_weights'][num_grad]
                    u_daniel = grad_aligned_sim_daniel[:, 0: num_grad + 1]
                    v_daniel = aligments_empirical[:, 0: num_grad + 1]
                    dict_simulations['Frob_grad_alig_daniel'][exper, num_grad][count_raw, count_col]  = np.average(np.mean((u_daniel - v_daniel)**2, axis = 0), weights= weights)
                    dict_simulations['Corr_grad_alig_daniel'][exper, num_grad][count_raw, count_col]  = np.average(1 + np.corrcoef(u_daniel, v_daniel, rowvar = False)[np.arange(0, num_grad + 1), np.arange(num_grad + 1, 2*(num_grad + 1))], weights = weights)
                    

                    u_emp = grad_aligned_sim_emp[:, 0: num_grad + 1]
                    v_emp = gradients_empirical[:, 0: num_grad + 1]
                    
                    dict_simulations['Frob_grad_alig_emp'][exper, num_grad][count_raw, count_col] = np.average(np.mean((u_emp - v_emp)**2, axis = 0), weights = weights)
                    dict_simulations['Corr_grad_alig_emp'][exper, num_grad][count_raw, count_col]  = np.average( 1 + np.corrcoef(u_emp, v_emp, rowvar = False)[np.arange(0, num_grad + 1), np.arange(num_grad + 1,  2*(num_grad + 1))],weights = weights)
                    
                    if num_grad > 5 :
                        dict_simulations['SSIM_grad_alig_daniel'][exper, num_grad][count_raw, count_col] = 1 - ssim(u_daniel, v_daniel, data_range=2)
                        dict_simulations['SSIM_grad_alig_emp'][exper, num_grad][count_raw, count_col]  = 1 - ssim(u_emp, v_emp, data_range=2)



    
            
            if count_col != len(G_all) -1:
                count_col += 1
            else:
                count_col = 0
        count_raw += 1


# plt.imshow(matrices_mean_FC_corr[0])
# plt.colorbar()

# num_simulation = 4
# path = f"D:\Juan\Facultad\Doctorado\Gradients\Simulation_{num_simulation}"
 
# # checking if the directory demo_folder  
# # exist or not. 
# if not os.path.exists(path): 
      
#     # if the demo_folder directory is not present  
#     # then create it. 
#     os.makedirs(path)

# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FCD_{condition}.npy", matrices_mean_FCD)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FCD_{condition}.npy", matrices_mean_FCD_frob)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FC_{condition}.npy", matrices_mean_FC)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FC_{condition}.npy", matrices_mean_FC)

# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\KS_FCD_{condition}.npy", matrices_mean_FCD_KS)


# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_SSIM_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Frob_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Corr_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Cosine_1)


# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_SSIM_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Frob_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Corr_1)
# np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Cosine_1)




# plt.plot(G_all, matrices_mean_gradient_SSIM_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\SSIM_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Frob_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Frob_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Corr_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Corr_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Cosine_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Cosdist_1grad_aconst.png")
# plt.close()

#%%

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
from scipy.io import loadmat
from scipy.spatial import distance
from brainspace.gradient.alignment import procrustes_alignment
from scipy import stats




#load all needed data for the Simulation 
data_SC = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\Data_for_hopf_1000")
data_freq = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\Car\hopf_freq_SCH1000_RS")
data_empirical = loadmat(r"D:\Juan\Facultad\Doctorado\Gradients\empirical_observables_gradients_concat")

# Import reference gradient from CSV
reference = np.loadtxt(r"D:\Juan\Facultad\Doctorado\Gradients\Car\margulies_grads_schaefer1000.csv", delimiter=',')
reference = reference.T
reference10 = reference[:,0:10]


gradient_in_euclidean(aligments_empirical[:, 0:2])
plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\Results\Concat vs Non-Concat\grad_euclidean_emp.png")


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

lambdas_empirical_3 = lambdas_empirical[0, 0:3]/lambdas_empirical[0, 0:3].sum()

grad_emp_weight  =  gradients_empirical[:, 0:3]@(lambdas_empirical_3*np.identity(3))

SC = data_SC.get('SC', None)    #structure connectivity matrix

#q = np.quantile(SC[np.triu_indices(1000, k = 1)], q = 0.99)
#SC = (SC/q)*0.2

SC = (SC/SC.max())*0.2


numNodes = len(SC)  #number of nodes
freq = data_empirical.get(f'freq_{condition}', None)[0]  #frequencies of awake state
K = np.sum(SC, axis = 0) # constant to use in the models (coupling term)

tStep = 0.1


SC = SC + 1j*np.zeros_like(SC)

SC = np.array(SC, dtype = 'F')


#attributes of the Simulation Class (Hopf in this case)

flp = 0.008
fhi = 0.08

t_final = 2990
thermalization_time = 200
sim = Hopf_simulation(t1= 0, t2=t_final, t_thermal = thermalization_time, rescale_time=3, t_step = tStep, num_nodes = numNodes, integration_method = 'euler_maruyama', filter_band = (flp, fhi))
sim.FCD_parameters(60, 40, 3)


time = np.arange(0, t_final, tStep)


num_experiments = 1
matrices_mean_FCD = np.empty((num_experiments, len(FCD_empirical), len(FCD_empirical)))


a_all = np.arange(-0.1, 0.1, 0.01)
#a_all = [0.01]
G_all = np.arange(0, 3, 0.15)
#G_all = [0.75]


matrices_mean_FCD = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FC = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FC_frob = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_FCD_frob = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_FC_corr = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_FCD_KS = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_Synch = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_Metas = np.empty((num_experiments, len(a_all), len(G_all)))


matrices_mean_gradient_alig_daniel_SSIM_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Frob_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Corr_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Cosine_1 = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_gradient_alig_emp_SSIM_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Frob_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Corr_1 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_emp_Cosine_1 = np.empty((num_experiments, len(a_all), len(G_all)))

matrices_mean_gradient_alig_daniel_SSIM_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Frob_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Corr_3 = np.empty((num_experiments, len(a_all), len(G_all)))
matrices_mean_gradient_alig_daniel_Cosine_3 = np.empty((num_experiments, len(a_all), len(G_all)))



Grad = GradientMaps(approach='dm')


#FC_empirical = stats.zscore(FC_empirical)

subjects = 10

for exper in range(num_experiments):
    SSIM_FC = np.zeros((len(a_all), len(G_all)))
    SSIM_FCD = np.zeros((len(a_all), len(G_all)))
    frob_FC = np.zeros((len(a_all), len(G_all)))
    frob_FCD = np.zeros((len(a_all), len(G_all)))
    corr_FC = np.zeros((len(a_all), len(G_all)))
    
    KS = np.zeros((len(a_all), len(G_all)))
    Synch_metric = np.zeros((len(a_all), len(G_all)))
    Metas_metric = np.zeros((len(a_all), len(G_all)))
    
    SSIM_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_daniel_1 = np.zeros((len(a_all), len(G_all)))
    
    SSIM_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_emp_1 = np.zeros((len(a_all), len(G_all)))
    

    SSIM_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Frob_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Corr_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    Cos_grad_alig_daniel_3 = np.zeros((len(a_all), len(G_all)))
    

    count_raw = 0
    count_col = 0
    
    for a in a_all:
        
        for G in G_all:



            fcd = np.zeros(shape = (subjects, 9316))
            
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

                fcd[sub] = sim.FCD[np.triu_indices(137, k = 1)]
                FCs[sub] = sim.FC
                FCDs[sub] = sim.FCD
                #FC_temp[sub] =   FC_temp_cal(sim.t_series, time_window_duration = 240, time_window_overlap = 0 , tstep_of_data = 2 )
                
                metas.append(sim.Metastability)
                synch.append(sim.Synchronization)
                
                # SSIM_FCD_dict['SSIM'].append(1 - ssim(sim.FCD, FCD_empirical))
                # SSIM_FCD_dict['alpha'].append(alpha)
                # SSIM_FCD_dict['beta'].append(beta)
                # metric = Metrics(sim)
    
    
            # fisher_FC_temp = np.arctanh(FC_temp)
            # fisher_mean_FC_temp = np.mean(fisher_FC_temp, axis = 0)
            # FC_temp_mean = np.tanh(fisher_mean_FC_temp)
                
            
            fisher_FC = np.arctanh(FCs)
            fisher_FC_mean = np.mean(fisher_FC, axis = 0)
            
            FC_mean = np.tanh(fisher_FC_mean)
            
            #FC_mean = stats.zscore(FC_mean)
            
            fisher_FCD = np.arctanh(FCDs)
            fisher_FCD_mean = np.mean(fisher_FCD, axis = 0)
            
            FCD_mean = np.tanh(fisher_FCD_mean)

            runs = 5
            gradients = np.empty((runs, numNodes, 10))
            lambdas = np.empty((runs, numNodes, 10))
            grad_aligned_daniel = np.empty((runs, numNodes, 10))
            grad_aligned_emp = np.empty((runs, numNodes, 10))
            
            for run in range(runs):
                Gsim = Grad.fit(FC_mean)
                aligned_daniel = procrustes_alignment([reference10, Gsim.gradients_], n_iter = 30)
                aligned_emp = procrustes_alignment([gradients_empirical, Gsim.gradients_], n_iter = 30)                
                
                
                gradients[run] =  Gsim.gradients_
                lambdas[run] = Gsim.lambdas_
                grad_aligned_daniel[run] = aligned_daniel[1]
                grad_aligned_emp[run] = aligned_emp[1]
                
            gradients_sim = gradients.mean(axis = 0)
            lambdas_sim = lambdas.mean(axis = 0)
            grad_aligned_sim_daniel = grad_aligned_daniel.mean(axis = 0)
            grad_aligned_sim_emp = grad_aligned_emp.mean(axis = 0)
            #grad_aligned_sim = gradients_sim


            
            #fcd_total[count_raw, count_col] = fcd.flatten()
            
            # SSIM_FCt_temporal = []
            # frob_FCt_temporal = []
            # for i in range(2, 7):
            #     SSIM_FCt_temporal.append(1 - ssim(FC_temp_mean[i], FCt_empirical[i]))
            #     frob_FCt_temporal.append(np.linalg.norm(FC_temp_mean[i] - FCt_empirical[i])/np.linalg.norm(FCt_empirical[i]))
            
            # SSIM_FCt[count_raw, count_col] = np.mean(SSIM_FCt_temporal)
            # frob_FCt[count_raw, count_col] = np.mean(frob_FCt_temporal)

            
            SSIM_FC[count_raw, count_col] = 1 - ssim(FC_mean, FC_empirical, data_range=2)
            SSIM_FCD[count_raw, count_col] = 1 - ssim(FCD_mean, FCD_empirical,  data_range=2)
            frob_FCD[count_raw, count_col] = np.linalg.norm(FCD_mean - FCD_empirical)/np.linalg.norm(FCD_empirical)
            frob_FC[count_raw, count_col] = np.linalg.norm(FC_mean - FC_empirical)/np.linalg.norm(FC_empirical)
            corr_FC[count_raw, count_col] = (1 - np.corrcoef(FC_empirical[np.triu_indices(numNodes, k = 1)], FC_mean[np.triu_indices(numNodes, k = 1)])[0, 1])

            
            KS[count_raw, count_col] = stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
            Synch_metric[count_raw, count_col] =  np.abs((np.mean(synch) - Synchronization_empirical))/Synchronization_empirical
            Metas_metric[count_raw, count_col] = np.abs((np.mean(metas) - Metastability_empirical))/Metastability_empirical
            
            
            SSIM_grad_alig_daniel_1[count_raw, count_col] = 1 - ssim(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0], data_range=2)
        
            Frob_grad_alig_daniel_1[count_raw, count_col] = np.linalg.norm(grad_aligned_sim_daniel[:, 0] - aligments_empirical[:, 0])/np.linalg.norm(aligments_empirical[:, 0])
        
            Corr_grad_alig_daniel_1[count_raw, count_col] = np.corrcoef(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])[0, 1]
    
            Cos_grad_alig_daniel_1[count_raw, count_col] = distance.cosine(grad_aligned_sim_daniel[:, 0], aligments_empirical[:, 0])
            


            SSIM_grad_alig_emp_1[count_raw, count_col] = 1 - ssim(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0], data_range=2)
        
            Frob_grad_alig_emp_1[count_raw, count_col] = np.linalg.norm(grad_aligned_sim_emp[:, 0] - gradients_empirical[:, 0])/np.linalg.norm(gradients_empirical[:, 0])
        
            Corr_grad_alig_emp_1[count_raw, count_col] = np.corrcoef(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0])[0, 1]
    
            Cos_grad_alig_emp_1[count_raw, count_col] = distance.cosine(grad_aligned_sim_emp[:, 0], gradients_empirical[:, 0])
            
    

        
            # SSIM_grad_alig_daniel_3[count_raw, count_col] = 1 - ( lambdas_empirical_3[0]*ssim(grad_aligned_sim[:, 0], gradients_empirical[:, 0], data_range=2) + \
            #                                              lambdas_empirical_3[1]*ssim(grad_aligned_sim[:, 1], gradients_empirical[:, 1], data_range=2) + \
            #                                                  lambdas_empirical_3[2]*ssim(grad_aligned_sim[:, 2], gradients_empirical[:, 2], data_range=2) )
        
            
            # grad_sim_weight  =  grad_aligned_sim[:, 0:3]@(lambdas_empirical_3*np.identity(3))
        
            # Frob_grad_alig_daniel_3[count_raw, count_col] = np.linalg.norm(grad_sim_weight - grad_emp_weight)/np.linalg.norm(grad_emp_weight)
        
            # Corr_grad_alig_daniel_3[count_raw, count_col] = lambdas_empirical_3[0]*(np.corrcoef(grad_aligned_sim[:, 0], gradients_empirical[:, 0])[0, 1]) + \
            #                                        lambdas_empirical_3[1]*(np.corrcoef(grad_aligned_sim[:, 1], gradients_empirical[:, 1])[0, 1]) + \
            #                                         lambdas_empirical_3[2]*(np.corrcoef(grad_aligned_sim[:, 2], gradients_empirical[:, 2])[0, 1])
            
            # Cos_grad_alig_daniel_3[count_raw, count_col] = lambdas_empirical_3[0]*distance.cosine(grad_aligned_sim[:, 0], gradients_empirical[:, 0]) +\
            #                                         lambdas_empirical_3[1]*distance.cosine(grad_aligned_sim[:, 1], gradients_empirical[:, 1]) +\
            #                                             lambdas_empirical_3[2]*distance.cosine(grad_aligned_sim[:, 2], gradients_empirical[:, 2])
          
                    
            #KS_post[count_raw, count_col] = stats.ks_2samp(fcd.flatten(), FCD_hist_empirical)[0]
   
            
            print(a, G, SSIM_grad_alig_daniel_1[count_raw, count_col],  Frob_grad_alig_daniel_1[count_raw, count_col], Corr_grad_alig_daniel_1[count_raw, count_col])
    
            
            if count_col != len(G_all) -1:
                count_col += 1
            else:
                count_col = 0
        count_raw += 1
    matrices_mean_FCD[exper] = SSIM_FCD
    matrices_mean_FC[exper] = SSIM_FC
    matrices_mean_FCD_frob[exper] = frob_FCD
    matrices_mean_FC_frob[exper] = frob_FC   
    matrices_mean_FC_corr[exper] = corr_FC 
 
    
    matrices_mean_FCD_KS[exper]  = KS
    matrices_mean_Synch[exper]  = Synch_metric

    matrices_mean_Metas[exper]  = Metas_metric
    
    matrices_mean_gradient_alig_daniel_SSIM_1[exper]= SSIM_grad_alig_daniel_1

    matrices_mean_gradient_alig_daniel_Frob_1[exper] = Frob_grad_alig_daniel_1
    
    matrices_mean_gradient_alig_daniel_Corr_1[exper] = Corr_grad_alig_daniel_1

    matrices_mean_gradient_alig_daniel_Cosine_1[exper] = Cos_grad_alig_daniel_1
    
    
    matrices_mean_gradient_alig_emp_SSIM_1[exper]= SSIM_grad_alig_emp_1

    matrices_mean_gradient_alig_emp_Frob_1[exper] = Frob_grad_alig_emp_1
    
    matrices_mean_gradient_alig_emp_Corr_1[exper] = Corr_grad_alig_emp_1

    matrices_mean_gradient_alig_emp_Cosine_1[exper] = Cos_grad_alig_emp_1
    
    # matrices_mean_gradient_alig_daniel_SSIM_3[exper]= SSIM_grad_3

    # matrices_mean_gradient_alig_daniel_Frob_3[exper] = Frob_grad_3
    
    # matrices_mean_gradient_alig_daniel_Corr_3[exper] = Corr_grad_3

    # matrices_mean_gradient_alig_daniel_Cosine_3[exper] = Cos_grad_3


plt.imshow(SC_other - SC)
plt.colorbar()

num_simulation = 5
path = f"D:\Juan\Facultad\Doctorado\Gradients\Simulation_{num_simulation}"
 
# checking if the directory demo_folder  
# exist or not. 
if not os.path.exists(path): 
      
    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs(path)

np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FCD_{condition}.npy", matrices_mean_FCD)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FCD_{condition}.npy", matrices_mean_FCD_frob)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_FC_{condition}.npy", matrices_mean_FC)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_FC_{condition}.npy", matrices_mean_FC)

np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\KS_FCD_{condition}.npy", matrices_mean_FCD_KS)


np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_SSIM_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Frob_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Corr_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_daniel_{condition}.npy", matrices_mean_gradient_alig_daniel_Cosine_1)


np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\SSIM_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_SSIM_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Frob_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Frob_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Corr_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Corr_1)
np.save(fr"D:\Juan\Facultad\Doctorado\Gradients\Results\Simulation_{num_simulation}\Cos_grad_1st_emp_{condition}.npy", matrices_mean_gradient_alig_emp_Cosine_1)


plt.imshow(matrices_mean_FC)
plt.colorbar()

# plt.plot(G_all, matrices_mean_gradient_SSIM_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\SSIM_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Frob_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Frob_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Corr_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Corr_1grad_aconst.png")
# plt.close()

# plt.plot(G_all, matrices_mean_gradient_Cosine_1[0, 0])
# plt.savefig(r"D:\Juan\Facultad\Doctorado\Gradients\prueba\Cosdist_1grad_aconst.png")
# plt.close()