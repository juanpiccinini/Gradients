import pickle
from Simulations_yoni import *
import warnings
import os
import itertools



#aca se cabiaron los == por is antes del None



class Metrics:
    
    global data   
    
    data = loadmat(r"D:\Juan\Facultad\Doctorado\DMT\Datos Empiricos\empirical_observables_DMT_experiment.mat")
    
    # #data = loadmat("D:\Juan\Facultad\Doctorado\Ephaptic Coupling\MEG\Deco\empirical_rs_data.mat")
    
    # FC_empirical = data.get('FC', None)
    
    # FCD_hist_empirical = data.get('FCD_hist', None)[0]
    # #FCD_fourier_empirical = data.get('FCD_fourier', None)
    
    # empirical_synch = data.get('Synchronization', None)[0][0]


    # empirical_metastability = data.get('Metastability', None)[0][0]
    
       #assert 'FC' in data, "Empirical data has not FC attribute"

    def __init__(self, simulation):
        self.simulation = simulation
        # self.corr_FC = 0
        # self.SSIM_FC = 0
        # self.frob_FC = 0
        # self.metastability_dist = 0
        # self.synchronization_dist = 0
        # self.KS_FCD = 0     #kolmogorov-Smirnov distance
        # self.corr_ps_FCD = 0    #correlation of 2D fourier FCD
        # self.ps_FCD = 0   #power spectrum distance
        # self.dist_corr = 0

    
    def correlation_distance(self, observable):
        """"Measure the similarity between two matrices computing the Pearson correlation
        coefficient between corresponding elements of the upper triangular part of those matrices."""

        
        assert observable in data, print(f"'{observable}' is not an attribute of Empirical Data ")
        names = ['corr_', observable]
        name = ''.join(names)
        

        v1 = data[observable][np.triu_indices(num_elements, k = 1)]
        
        matrix = getattr(self.simulation, observable)
        v2 = matrix[np.triu_indices(num_elements, k = 1)]
      
        corr_dist = (1 - stats.pearsonr(v2, v1)[0])
        
        setattr(self, name, corr_dist)

        return corr_dist


    def SSIM_distance(self, observable):
        """returns the 1 - SSIM distance between simulated matrix
        and the empirical one."""

        assert observable in data, print(f"'{observable}' is not an attribute of Empirical Data ")
        names = ['SSIM_', observable]
        name = ''.join(names)
        

        v1 = data[observable]
        
        matrix = getattr(self.simulation, observable)
        v2 = matrix
      
        metric = 1- ssim(v2, v1)
        setattr(self, name, metric)

        return metric


    def frob_distance(self, observable):
        """returns the Frobenius norm of the difference between the simulated matrix
        and the empirical one."""
        
    
        assert observable in data, print(f"'{observable}' is not an attribute of Empirical Data ")
        names = ['frob_', observable]
        name = ''.join(names)
        

        v1 = data[observable]
        
        matrix = getattr(self.simulation, observable)
        v2 = matrix
      
        metric = np.linalg.norm(v2 - v1)/np.linalg.norm(v1)
        setattr(self, name, metric)
        
        return 
    

    def kolmogorov_distance(self, observable):
        
        
        assert observable in data, print(f"'{observable}' is not an attribute of Empirical Data ")
        names = ['ks_', observable]
        name = ''.join(names)
        
     

        matrix_emp = data[observable]
        num_dimensions = len(matrix_emp)
        
        
            
        
        matrix = getattr(self.simulation, observable)
        num_windows = len(matrix)   
        
        if num_dimensions == 3:
            v1 = np.array([matrix_temp[np.triu_indices(num_windows, k = 1)] for matrix_temp in matrix_emp]).flatten()

        else:
            v1 = matrix_emp[np.triu_indices(num_windows, k = 1)]
        
        v2 = matrix[np.triu_indices(num_windows, k = 1)]
      
        
        metric = stats.ks_2samp(v2, v1)
        setattr(self, name, metric)

        return metric

    def observable_distance(self, observable):
        """returns the relative distance between the simulated scalar observable
        and the empirical one."""

        
        assert observable in data, print(f"'{observable}' is not an attribute of Empirical Data ")
        names = [observable, '_distance']
        name = ''.join(names)
        
     

        emp = data[observable]
        
        simulated = getattr(self.simulation, observable)
        
        metric = abs((simulated - emp))/emp
        
        setattr(self, name, metric)
        
        return metric
    
    
   
    def SSIM_FCD_ps(self, num_subjects = None):
        """returns the 1 - SSIM distance between the simulated FC
        and the empirical one."""
        FCD_empirical_ps = self.FCD_fourier_empirical
        if num_subjects is None:
            fourier = self.simulation.FCD_ps
        else:
           fourier = self.simulation.FCD_ps/num_subjects 
        
        simulation_log = np.log10(fourier + 1)
        empirical_log = np.log10(FCD_empirical_ps + 1)
        return 1 - ssim(simulation_log, empirical_log)
    
    def SSIM_FCD_ps_old(self):
        """returns the 1 - SSIM distance between the simulated FC
        and the empirical one."""
        FCD_empirical_ps = self.FCD_fourier_empirical
        fourier = sp.fft_2d(self.simulation.FCD)
        return 1 - ssim(fourier, FCD_empirical_ps)

    def frob_FCD_ps(self, num_subjects = None):
        """returns the Frobenius norm of the difference between the power 
        spectrum of the simulated FCD and the empirical one."""
        FCD_empirical_ps = self.FCD_fourier_empirical
        if num_subjects is None:
            fourier = self.simulation.FCD_ps
        else:
           fourier = self.simulation.FCD_ps/num_subjects 

        return np.linalg.norm(fourier - FCD_empirical_ps)/np.linalg.norm(FCD_empirical_ps)
    
    def correlation_FCD(self, num_subjects = None):
        """"Measure the similarity between two matrices computing the Pearson correlation
        coefficient between corresponding elements of the upper triangular part of those matrices."""
        FCD_empirical_ps = self.FCD_fourier_empirical
        if num_subjects == None:
            fourier = self.simulation.FCD_ps
        else:
           fourier = self.simulation.FCD_ps/num_subjects 
        num_elements = len(FCD_empirical_ps)
        return (1 - stats.pearsonr(fourier[np.triu_indices(num_elements, k = 1)], FCD_empirical_ps[np.triu_indices(num_elements, k = 1)])[0])

    def distance_corr(self, num_subjects = None):
        """"Measure the similarity between two matrices computing the Pearson correlation
        coefficient between corresponding elements of the upper triangular part of those matrices."""
        num_elements = len(self.FC_empirical)
        if num_subjects is None:
            v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
            v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]
    
        else: 
            v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
            v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]/num_subjects
        return (1 - stats.pearsonr(v2, v1)[0])




# class Metrics:
    
        
#     data = loadmat('empirical_observables_awake.mat')
#     #data = loadmat("D:\Juan\Facultad\Doctorado\Ephaptic Coupling\MEG\Deco\empirical_rs_data.mat")
    
#     FC_empirical = data.get('FC', None)
    
#     FCD_hist_empirical = data.get('FCD_hist', None)[0]
#     #FCD_fourier_empirical = data.get('FCD_fourier', None)
    
#     empirical_synch = data.get('Synchronization', None)[0][0]


#     empirical_metastability = data.get('Metastability', None)[0][0]
    
#     #global data
#     #assert 'FC' in data, "Empirical data has not FC attribute"

#     def __init__(self, simulation):
#         self.simulation = simulation
#         self.corr_FC = 0
#         self.SSIM_FC = 0
#         self.frob_FC = 0
#         self.metastability_dist = 0
#         self.synchronization_dist = 0
#         self.KS_FCD = 0     #kolmogorov-Smirnov distance
#         self.corr_ps_FCD = 0    #correlation of 2D fourier FCD
#         self.ps_FCD = 0   #power spectrum distance
#         self.dist_corr = 0

#     def correlation_FC(self, num_subjects = None):
#         """"Measure the similarity between two matrices computing the Pearson correlation
#         coefficient between corresponding elements of the upper triangular part of those matrices."""
#         num_elements = len(self.FC_empirical)
#         if num_subjects is None:
#             v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
#             v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]
    
#         else: 
#             v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
#             v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]/num_subjects
#         return (1 - stats.pearsonr(v2, v1)[0])

#     def SSIM_distance(self, num_subjects = None):
#         """returns the 1 - SSIM distance between the simulated FC
#         and the empirical one."""
#         if num_subjects is None:
#             FC_sim = self.simulation.FC
    
#         else: 
#             FC_sim = self.simulation.FC/num_subjects
        
#         return 1- ssim(FC_sim, self.FC_empirical)


#     def frob_distance(self, num_subjects = None):
#         """returns the Frobenius norm of the difference between the simulated FC
#         and the empirical one."""
#         if num_subjects == None:
#             FC_sim = self.simulation.FC
    
#         else: 
#             FC_sim = self.simulation.FC/num_subjects
        
#         return np.linalg.norm(FC_sim - self.FC_empirical)/np.linalg.norm(self.FC_empirical)
    

#     def kolmogorov_distance(self, num_subjects = None):

#         FCD_array_empirical = self.FCD_hist_empirical
#         if num_subjects == None:
#             num_windows = len(self.simulation.FCD)
#             FCD_array_simulated = self.simulation.FCD[np.triu_indices(num_windows, k = 1)]
#         else:
#             FCD_array_simulated = self.simulation.FCD
#         return stats.ks_2samp(FCD_array_simulated, FCD_array_empirical)

#     def metastability_distance(self, num_subjects = None):
#         """returns the relative distance between the simulated metastability
#         and the empirical one."""
#         if num_subjects is None:
#             metastability = self.simulation.Metastability
#         else:
#            metastability = self.simulation.Metastability/num_subjects 
        
#         return abs((metastability - self.empirical_metastability))/self.empirical_metastability
    
    
#     def synchronization_distance(self, num_subjects = None):
#         """returns the relative distance between the simulated synchronization
#         and the empirical one."""
#         if num_subjects is None:
#             synch = self.simulation.Synchronization
#         else:
#            synch = self.simulation.Synchronization/num_subjects 
        
#         return (synch - self.empirical_synch)/self.empirical_synch 
   
#     def SSIM_FCD_ps(self, num_subjects = None):
#         """returns the 1 - SSIM distance between the simulated FC
#         and the empirical one."""
#         FCD_empirical_ps = self.FCD_fourier_empirical
#         if num_subjects is None:
#             fourier = self.simulation.FCD_ps
#         else:
#            fourier = self.simulation.FCD_ps/num_subjects 
        
#         simulation_log = np.log10(fourier + 1)
#         empirical_log = np.log10(FCD_empirical_ps + 1)
#         return 1 - ssim(simulation_log, empirical_log)
    
#     def SSIM_FCD_ps_old(self):
#         """returns the 1 - SSIM distance between the simulated FC
#         and the empirical one."""
#         FCD_empirical_ps = self.FCD_fourier_empirical
#         fourier = sp.fft_2d(self.simulation.FCD)
#         return 1 - ssim(fourier, FCD_empirical_ps)

#     def frob_FCD_ps(self, num_subjects = None):
#         """returns the Frobenius norm of the difference between the power 
#         spectrum of the simulated FCD and the empirical one."""
#         FCD_empirical_ps = self.FCD_fourier_empirical
#         if num_subjects is None:
#             fourier = self.simulation.FCD_ps
#         else:
#            fourier = self.simulation.FCD_ps/num_subjects 

#         return np.linalg.norm(fourier - FCD_empirical_ps)/np.linalg.norm(FCD_empirical_ps)
    
#     def correlation_FCD(self, num_subjects = None):
#         """"Measure the similarity between two matrices computing the Pearson correlation
#         coefficient between corresponding elements of the upper triangular part of those matrices."""
#         FCD_empirical_ps = self.FCD_fourier_empirical
#         if num_subjects == None:
#             fourier = self.simulation.FCD_ps
#         else:
#            fourier = self.simulation.FCD_ps/num_subjects 
#         num_elements = len(FCD_empirical_ps)
#         return (1 - stats.pearsonr(fourier[np.triu_indices(num_elements, k = 1)], FCD_empirical_ps[np.triu_indices(num_elements, k = 1)])[0])

#     def distance_corr(self, num_subjects = None):
#         """"Measure the similarity between two matrices computing the Pearson correlation
#         coefficient between corresponding elements of the upper triangular part of those matrices."""
#         num_elements = len(self.FC_empirical)
#         if num_subjects is None:
#             v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
#             v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]
    
#         else: 
#             v1 = self.FC_empirical[np.triu_indices(num_elements, k = 1)]
#             v2 = self.simulation.FC[np.triu_indices(num_elements, k = 1)]/num_subjects
#         return (1 - stats.pearsonr(v2, v1)[0])

    
#     def fitting(self, num_subjects):
#         self.corr_FC = self.correlation_FC(num_subjects = num_subjects)
#         self.SSIM_FC = self.SSIM_distance(num_subjects= num_subjects)
#         self.frob_FC = self.frob_distance(num_subjects = num_subjects)
#         self.metastability_dist = self.metastability_distance(num_subjects = num_subjects)
#         self.synchronization_dist = self.synchronization_distance(num_subjects = num_subjects)
#         self.KS_FCD = self.kolmogorov_distance(num_subjects = num_subjects)[0]     #kolmogorov-Smirnov distance
#         self.corr_ps_FCD = self.correlation_FCD(num_subjects = num_subjects)    #correlation of 2D fourier FCD
#         self.ps_FCD = self.frob_FCD_ps(num_subjects = num_subjects)   #power spectrum distance



#         return self.corr_FC, self.SSIM_FC, self.frob_FC, self.metastability_dist,\
#                  self.synchronization_dist, self.KS_FCD, self.corr_ps_FCD, self.ps_FCD 
    




def loop_a(sim, distances, models, model, parameters, directory):
    distances = distances
    index_1 = 0
    warnings.filterwarnings("error")
    for i in range(len(parameters['a'])):
        index_2 = 0
        for j in range(len(parameters['coupling'])):
            a, b, c, G = parameters['a'][i], parameters['b'][0], parameters['c'][0], parameters['coupling'][j]
            sim.rossler_parameters( a, b, c, G)
            try:
                print(a, b, c, G)
                sim.run_sim()
                file_name = directory + os.sep + 'Rossler_{0}_{1}_{2}_{3}'.format(a, b, c, G) +  '.' +  'pkl'
                with open(file_name, 'wb') as output:
                    pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
    
                corr_FC, SSIM_FC,frob_FC, metastability_dist,\
                synchronization_dist, KS_FCD, corr_ps_FCD, ps_FCD  = Metrics(sim).fitting()
                distances[0][index_1][index_2] = corr_FC
                distances[1][index_1][index_2] = SSIM_FC
                distances[2][index_1][index_2] = frob_FC
                distances[3][index_1][index_2] = metastability_dist
                distances[4][index_1][index_2] = synchronization_dist
                distances[5][index_1][index_2] = KS_FCD
                distances[6][index_1][index_2] = corr_ps_FCD
                distances[7][index_1][index_2] = ps_FCD
            except RuntimeWarning:
                distances[0][index_1][index_2]  = np.nan
                distances[1][index_1][index_2]  = np.nan
                distances[2][index_1][index_2]  = np.nan
                distances[3][index_1][index_2]  = np.nan
                distances[4][index_1][index_2]  = np.nan
                distances[5][index_1][index_2]  = np.nan
                distances[6][index_1][index_2]  = np.nan
                distances[7][index_1][index_2]  = np.nan
                pass
            index_2 += 1
        index_1 += 1
    return distances

def loop_b(sim, distances, models, model, parameters, directory):
    distances = distances
    index_1 = 0
    warnings.filterwarnings("error")
    for i in range(len(parameters['b'])):
        index_2 = 0
        for j in range(len(parameters['coupling'])):
            a, b, c, G = parameters['a'][0], parameters['b'][i], parameters['c'][0], parameters['coupling'][j]
            sim.rossler_parameters( a, b, c, G)
            try:
                print(a, b, c, G)
                sim.run_sim()
                file_name = directory + os.sep + 'Rossler_{0}_{1}_{2}_{3}'.format(a, b, c, G) +  '.' +  'pkl'
                with open(file_name, 'wb') as output:
                    pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
    
                corr_FC, SSIM_FC,frob_FC, metastability_dist,\
                synchronization_dist, KS_FCD, corr_ps_FCD, ps_FCD  = Metrics(sim).fitting()
                distances[0][index_1][index_2] = corr_FC
                distances[1][index_1][index_2] = SSIM_FC
                distances[2][index_1][index_2] = frob_FC
                distances[3][index_1][index_2] = metastability_dist
                distances[4][index_1][index_2] = synchronization_dist
                distances[5][index_1][index_2] = KS_FCD
                distances[6][index_1][index_2] = corr_ps_FCD
                distances[7][index_1][index_2] = ps_FCD
            except RuntimeWarning:
                distances[0][index_1][index_2]  = np.nan
                distances[1][index_1][index_2]  = np.nan
                distances[2][index_1][index_2]  = np.nan
                distances[3][index_1][index_2]  = np.nan
                distances[4][index_1][index_2]  = np.nan
                distances[5][index_1][index_2]  = np.nan
                distances[6][index_1][index_2]  = np.nan
                distances[7][index_1][index_2]  = np.nan
                pass
            index_2 += 1
        index_1 += 1
    return distances

def loop_c(sim, distances, models, model, parameters, directory):
    distances = distances
    index_1 = 0
    warnings.filterwarnings("error")
    for i in range(len(parameters['c'])):
        index_2 = 0
        for j in range(len(parameters['coupling'])):
            a, b, c, G = parameters['a'][0], parameters['b'][0], parameters['c'][i], parameters['coupling'][j]
            sim.rossler_parameters( a, b, c, G)
            try:
                print(a, b, c, G)
                sim.run_sim()
                file_name = directory + os.sep + 'Rossler_{0}_{1}_{2}_{3}'.format(a, b, c, G) +  '.' +  'pkl'
                with open(file_name, 'wb') as output:
                    pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
    
                corr_FC, SSIM_FC,frob_FC, metastability_dist,\
                synchronization_dist, KS_FCD, corr_ps_FCD, ps_FCD  = Metrics(sim).fitting()
                distances[0][index_1][index_2] = corr_FC
                distances[1][index_1][index_2] = SSIM_FC
                distances[2][index_1][index_2] = frob_FC
                distances[3][index_1][index_2] = metastability_dist
                distances[4][index_1][index_2] = synchronization_dist
                distances[5][index_1][index_2] = KS_FCD
                distances[6][index_1][index_2] = corr_ps_FCD
                distances[7][index_1][index_2] = ps_FCD
            except RuntimeWarning:
                distances[0][index_1][index_2]  = np.nan
                distances[1][index_1][index_2]  = np.nan
                distances[2][index_1][index_2]  = np.nan
                distances[3][index_1][index_2]  = np.nan
                distances[4][index_1][index_2]  = np.nan
                distances[5][index_1][index_2]  = np.nan
                distances[6][index_1][index_2]  = np.nan
                distances[7][index_1][index_2]  = np.nan
                pass
            index_2 += 1
        index_1 += 1
    return distances


# def distances_2d(directory, **kwargs):
        
#     directory = directory
#     scalars = {}
#     arrays = {}
#     for key, value in kwargs.items():
#         val = np.isscalar(value)
#         if val == True:
#             scalars[key] = value
#         else:
#             arrays[key] = value
        
#     for key, value in scalars.items():
#         if key == 'a':
#             a = value
#         if key == 'b':
#             b = value
#         if key == 'c':
#             c = value
    
#     parameter_1 = []
#     parameter_2 = []
#     for key, value in arrays.items(): 
#         if key == 'a':
#             parameter_1 = arrays['a']
#             parameter_2 = arrays['G']
#         if key == 'b':
#             parameter_1 = arrays['b']
#             parameter_2 = arrays['G']
#         if key == 'c':
#             parameter_1 = arrays['c']
#             parameter_2 = arrays['G']  
    
#     print(parameter_1, parameter_2)
    
#     warnings.filterwarnings("error")
#     distances = np.zeros(shape=(4, len(parameter_1), len(parameter_2)))
#     index_1 = 0
#     for i in parameter_1:
#         print('The 1st parameter has started = ', i)
#         index_2 = 0
#         for G in parameter_2:
#             print('The 2nd parameter has started =', G)
#             tStep = 0.1
#             sim = Rossler_simulation(t1= 0, t2=7000, t_thermal=1000, rescale_time=2, t_step=tStep)
#             sim.ini_conditions(0.1, 0.1, 0.1)
#             sim.FCD_parameters(60, 40, 2)
#             for key, value in arrays.items(): 
#                 if key == 'a':
#                     a = i
#                     sim.rossler_parameters(a, b, c, G)
#                 if key == 'b':
#                     b = i
#                     sim.rossler_parameters(a, b, c, G)
#                 if key == 'c':
#                     c = i
#                     sim.rossler_parameters(a, b, c, G)
#             try:
#                 print(a, b, c, G)
#                 sim.run_sim()
#                 file_name = directory + os.sep + 'Rossler {0} {1} {2} {3}'.format(a, b, c, G) +  '.' +  'pkl'
#                 with open(file_name, 'wb') as output:
#                     pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
    
#                 FC_dist, FCD_KS, FCD_ps_dist, metastability_dist = Metrics(sim).fitting()
#                 distances[0][index_1][index_2] = FC_dist
#                 distances[1][index_1][index_2] = FCD_KS
#                 distances[2][index_1][index_2] = FCD_ps_dist
#                 distances[3][index_1][index_2] = metastability_dist
#             except RuntimeWarning:
#                 print("Los parametros a = {}, b = {}, c = {}, G = {} son una verga!!!".format(a, b, c, G))
#                 distances[0][index_1][index_2]  = np.nan
#                 distances[1][index_1][index_2]  = np.nan
#                 distances[2][index_1][index_2]  = np.nan
#                 distances[3][index_1][index_2]  = np.nan
#                 pass
    
#             index_2 += 1
        
#         index_1 += 1
#     return distances



def calculator(model, num_runs,  directory, **kwargs):
   
    directory = directory
    models = {'rossler': Rossler_simulation, 'hopf': Hopf_simulation}
    parameters = {}
    iterators = []
    for key, value in kwargs.items():
        val = np.isscalar(value)
        if val == True:
            parameters[key] = [value]
        else:
            parameters[key] = value

            iterators.append(key)
    print(iterators)
    for a, b in itertools.permutations(iterators):
        if a == 'coupling':
            iterators = [b, a]
        else:
            pass
    warnings.filterwarnings("error")
    
    
    if model == 'rossler':
        tStep = 0.1
        loops = {'a': loop_a, 'b': loop_b, 'c': loop_c}
        distances = np.zeros(shape = (8, len(parameters[iterators[0]]), len(parameters[iterators[1]]) ))
        sim = Rossler_simulation(t1= 0, t2=4000, t_thermal=1000, rescale_time=2, t_step=tStep)
        sim.ini_conditions(0.1, 0.1, 0.1)
        sim.FCD_parameters(60, 40, 2)
        print(loops.get(iterators[0]))
        loops.get(iterators[0])(sim, distances, models, model, parameters, directory)

    
    elif model == 'hopf':
        tStep = 0.1
        distances = np.zeros(shape = (8, len(parameters[iterators[0]]), len(parameters[iterators[1]]) ))
        index_1 = 0
        warnings.filterwarnings("error")
        sim = Hopf_simulation(t1= 0, t2=4000, t_thermal=1000, rescale_time=2, t_step = tStep)
        sim.ini_conditions(0.1, 0.1)
        sim.FCD_parameters(60, 40, 2)
        for i in range(len(parameters['a'])):
            index_2 = 0
            for j in range(len(parameters['coupling'])):
                a, G = parameters['a'][i], parameters['coupling'][j]
                sim.hopf_parameters(a, G, 0.04*np.sqrt(tStep))
                print(a, G)
                for run in range(num_runs):
                    print('run number:', run)
                    folder = directory + os.sep + 'run_{0}'.format(run)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
           
                    sim.run_sim()
                    
                    file_name = folder + os.sep + 'Hopf {0} {1}'.format(a, G) +  '.' +  'pkl'
                    with open(file_name, 'wb') as output:
                        pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
                
                index_2 += 1
            index_1 += 1

    print('All work Done')


def distances_2d_2(model, num_runs,  directory, **kwargs):
   
    directory = directory
    models = {'rossler': Rossler_simulation, 'hopf': Hopf_simulation}
    parameters = {}
    iterators = []
    for key, value in kwargs.items():
        val = np.isscalar(value)
        if val == True:
            parameters[key] = [value]
        else:
            parameters[key] = value

            iterators.append(key)
    print(iterators)
    for a, b in itertools.permutations(iterators):
        if a == 'coupling':
            iterators = [b, a]
        else:
            pass
    warnings.filterwarnings("error")
    
    
    if model == 'rossler':
        tStep = 0.1
        loops = {'a': loop_a, 'b': loop_b, 'c': loop_c}
        distances = np.zeros(shape = (8, len(parameters[iterators[0]]), len(parameters[iterators[1]]) ))
        sim = Rossler_simulation(t1= 0, t2=4000, t_thermal=1000, rescale_time=2, t_step=tStep)
        sim.ini_conditions(0.1, 0.1, 0.1)
        sim.FCD_parameters(60, 40, 2)
        print(loops.get(iterators[0]))
        loops.get(iterators[0])(sim, distances, models, model, parameters, directory)

    
    elif model == 'hopf':
        tStep = 0.1
        distances = np.zeros(shape = (8, len(parameters[iterators[0]]), len(parameters[iterators[1]]) ))
        index_1 = 0
        warnings.filterwarnings("error")
        sim = Hopf_simulation(t1= 0, t2=4000, t_thermal=1000, rescale_time=2, t_step = tStep)
        sim.ini_conditions(0.1, 0.1)
        sim.FCD_parameters(60, 40, 2)
        for i in range(len(parameters['a'])):
            index_2 = 0
            for j in range(len(parameters['coupling'])):
                a, G = parameters['a'][i], parameters['coupling'][j]
                sim.hopf_parameters(a, G, 0.04*np.sqrt(tStep))
                print(a, G)
                subjects = dict()
                for run in range(num_runs):
                    print('run number:', run)
                    folder = directory + os.sep + 'run_{0}'.format(run)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
           
                    sim.run_sim()
                    subjects[run] = sim
                    
                    file_name = folder + os.sep + 'Hopf_{0}_{1}'.format(a, G) +  '.' +  'pkl'
                    with open(file_name, 'wb') as output:
                        pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)
                
                sim_total = subjects[0]
                num_win = len(subjects[0].FCD)
                FCD_array = sim_total.FCD[np.triu_indices(num_win, k = 1)]
                for subj in range(1, num_runs):
                    FCD_array_2 = subjects[subj].FCD[np.triu_indices(num_win, k = 1)]
                    FCD_array = np.concatenate((FCD_array, FCD_array_2))
                    sim_total += subjects[subj]
                sim_total.FCD = FCD_array
                print(Metrics(sim_total).synchronization_distance(num_subjects = num_runs ))
                corr_FC, SSIM_FC, frob_FC, metastability_dist,\
                synchronization_dist, KS_FCD, corr_ps_FCD, ps_FCD  = Metrics(sim_total).fitting(num_subjects = num_runs )
                print(a, G, synchronization_dist)
                distances[0][index_1][index_2] = corr_FC
                distances[1][index_1][index_2] = SSIM_FC
                distances[2][index_1][index_2] = frob_FC
                distances[3][index_1][index_2] = metastability_dist
                distances[4][index_1][index_2] = synchronization_dist
                distances[5][index_1][index_2] = KS_FCD
                distances[6][index_1][index_2] = corr_ps_FCD
                distances[7][index_1][index_2] = ps_FCD
                print(distances[4])
                index_2 += 1
            index_1 += 1

    return distances





