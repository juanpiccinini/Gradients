import numpy as np
from scipy.io import loadmat
from models import *
from Functional_Connectivity import *
from Kuramoto import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import operator
import pandas as pd
import signal_process as sp
from scipy.signal import butter, sosfilt, welch
from scipy.fftpack import fft2,fftshift
from time import process_time
import pickle
from ode_integrator import *
from scipy.integrate import solve_ivp
from ode_integrator import integration_ode_5
from BOLD_function import BOLD


class Simulation:


    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, filter_band = (0.04, 0.07)):
        print('Processing...')
        self.t1 = t1    #start of the time interval
        self.t2 = t2  #end of the time interval
        self.t_thermal = t_thermal   #themalization time
        self.rescale_time = rescale_time    #to make the time series compatible with the empirical data
        self.t_step = t_step     #lenght of the time step for integration
        self.num_nodes = num_nodes
        self.num_steps = int((t2-t1)/t_step) #number of step
        self.final_points = int((t2-t_thermal)/rescale_time)   #el calculo es equivalente a el slicing hecho mas abajo
                                                                      #para determinados valores de h, rescale_time, y t2
        self.Ny_freq = 0.5*(self.final_points/self.t2)
        self.filter_band = filter_band   #tuple of the frequency band requiered

    def FCD_parameters(self, time_window_duration, time_window_overlap, tstep_of_data):
        #parameters for calculating the FCD
        self.time_window_duration = time_window_duration
        self.time_window_overlap = time_window_overlap
        self.tstep_of_data = tstep_of_data
        self.num_windows = int((self.final_points*self.tstep_of_data - self.time_window_duration)
                          /(self.time_window_duration-self.time_window_overlap))
    
    
    def model_parameters(self, **kwargs):
        self.parameters = kwargs
        
    def initial_conditions(self, ic):
        """ic must be a matrix of type (num_dimensions, num_nodes)"""
        self.ic = ic


    def integrate_model(self,  model, method = 'runge_kutta2'):
        print('The integration is about to start!')

        t0 = process_time() 

        X = integration_ode(model, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step, method = method,  **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)
        
        return X
         


class Rossler_simulation(Simulation):
    
    num_dimensions = 3  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method, filter_band):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes, filter_band)
        self.integration_method = integration_method
   
    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        xpoints, ypoints, zpoints = integration_ode_2(rossler_brain, 
                            initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)

        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')
        
class Hopf_simulation(Simulation):
    
    num_dimensions = 1  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method, filter_band):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes, filter_band)
        self.integration_method = integration_method

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 


        zpoints = integration_ode_7(hopf_brain_faster, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)
        
        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        self.x = zpoints.real
        self.y = zpoints.imag
        
        #simulated data compatible with the empirical data
        xpoints_new = self.x[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        
        x_signal_no_filter = np.empty_like(xpoints_new)
        x_signal = np.empty_like(xpoints_new)
        
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
        
            if self.filter_band != None:
                sos = butter(2, [self.filter_band[0]/self.Ny_freq, self.filter_band[1]/self.Ny_freq], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            
            x_signal[:, node] = signal_2
            
            #signal without filtering to calculate the FCD
            x_signal_no_filter[::, node] = signal
            
        self.t_series = x_signal
        
        self.FC = FC_cal(x_signal)

        phase = phases(x_signal)

        self.phase = phase
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        #self.leida = Leida(phase)
        
        # Uhat, Shat, Vhat = np.linalg.svd(phase, full_matrices=False)
        # self.Uhat = Uhat
        # self.Shat = Shat
        # self.Vhat = Vhat
        
        t2 = process_time() 
        self.FCD = FCD_cal(x_signal_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')
        
class HopfEphap_simulation(Simulation):
    
    num_dimensions = 1  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method, filter_band, distance_matrix = None, NR = None, bold = 'no'):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes, filter_band)
        self.integration_method = integration_method
        self.distance_matrix = distance_matrix
        self.NR = NR
        self.bold = bold
        

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 


        zpoints = integration_ode_6(hopf_brain_ephaptic, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        xpoints = zpoints.real
   
        self.x = zpoints.real
        self.y = zpoints.imag
        
        t0_bold = process_time() 
        if self.bold == 'no':
            pass
        else:

            xpoints = xpoints.transpose()

            b = BOLD(self.t2 - self.t1, xpoints, self.t_step)
            xpoints = b.transpose()
            self.Bold_tseries = xpoints
        t1_bold = process_time() 
        print('BOLD time:', t1_bold - t0_bold)
        
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
     
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', fs = 1/self.rescale_time, output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)

        
        t_struc_init = process_time() 
        
        
        if self.NR is not None:
        
            max_dist = np.max(self.distance_matrix)
            delta = max_dist/self.NR
    
            index_matrix = np.floor(self.distance_matrix/delta).astype(int)
            index_uniq = np.unique(index_matrix)
    
            corrfcnt = np.zeros(shape = (self.num_nodes, len(index_uniq)))
            
            # for count_i, i in enumerate(index_uniq[1:]):
            
            #     num_index_temp = np.sum(index_matrix == i, axis = 1)
            #     matrix_bin = (index_matrix == i)*1
            #     numind_sparse = (num_index_temp*matrix_bin.T).T
            #     with np.errstate(divide='ignore'):  
            #         numind_matrix_inv = np.where(numind_sparse == 0, 0, 1/numind_sparse)
                
            
                
            #     corrfcnt[:, count_i] = np.sum(numind_matrix_inv*self.FC, axis = 1)
                
                
            self.corrfcn = corrfcnt 
                
        else: pass

        
        t_struc_final = process_time() 
        print("Time elapsed in B(t) calculation ", t_struc_final - t_struc_init)
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)
        
        t3 = process_time() 
        
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')  
        	
	


class HopfDelay_simulation(Simulation):
    
    num_dimensions = 1  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method, filter_band):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes, filter_band)
        self.integration_method = integration_method
        

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        zpoints = integration_ode_4(hopf_brain_delay, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        xpoints = zpoints.real
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        
        
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')  
        


class HopfEphap_simulation_new(Simulation):
    
    num_dimensions = 1  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes)
        self.integration_method = integration_method
    
        

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 
        
        dictionario = self.parameters
        lista = []
        for key in dictionario:
            lista.append(dictionario[key])

        parameters_tuple = tuple(lista)
        
        
        tspan = np.arange(self.t1, self.t2, self.t_step)
        y0 = self.ic.reshape(self.num_nodes)
        
        Z = solve_ivp(hopf_brain_ephaptic_2, t_span = (tspan[0], tspan[-1]), y0 = y0, args = (parameters_tuple), t_eval=tspan, method = 'BDF', max_step= self.t_step, atol = 1, rtol = 1)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)
        
        zpoints = Z.y.T
        xpoints = zpoints.real
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        
        
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')  
    

class BT_extended_simulation(Simulation):
    
    num_dimensions = 2  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes)
        self.integration_method = integration_method

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        xpoints, ypoints = integration_ode_3(bogdanov_takens_extended, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        
        
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished') 
        

class Norm_Hopf_simulation(Simulation):
    
    num_dimensions = 2  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes)
        self.integration_method = integration_method

    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        rhoPoints, thetaPoints = integration_ode_3(new_model, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        xpoints = rhoPoints*np.cos(thetaPoints)
        

        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.phase = phase
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        
        
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')  
        
        
class Hopf_extended_simulation(Simulation):
    
    num_dimensions = 2  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes)
        self.integration_method = integration_method

    
    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        xpoints, ypoints = integration_ode_3(Hopf_extended_2, initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)
        
        
        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)
        
        self.centerMass = np.array([np.mean(xpoints, axis = 0), np.mean(ypoints, axis = 0)])
        
        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished') 
        return self


class NormalF_3D_simulation(Simulation):
    
    num_dimensions = 3  #number of dimension of each node
    
    def __init__(self, t1, t2, t_thermal, rescale_time, t_step, num_nodes, integration_method):
        super().__init__( t1, t2, t_thermal, rescale_time, t_step, num_nodes)
        self.integration_method = integration_method
   
    def run_sim(self):
        
        print('The simulation is about to start!')
        FC = np.zeros(shape=(self.num_nodes, self.num_nodes))
        FCD = np.zeros(shape=(self.num_windows, self.num_windows))
        t_series = np.zeros(shape=(self.final_points, self.num_nodes))
        kuramoto = np.zeros(shape=(1 , self.final_points))
        
        t0 = process_time() 

        xpoints, ypoints, zpoints = integration_ode_2(new_model_3d, 
                            initial_conditions = self.ic, tmin = self.t1, tmax = self.t2,
                            tstep = self.t_step,  method = self.integration_method, **self.parameters)

        t1 = process_time() 
        print("Time elapsed in integration ", t1 - t0) # CPU seconds elapsed (floating point)

        
        #simulated data compatible with the empirical data
        xpoints_new = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        xpoints_new_no_filter = xpoints[int(self.t_thermal/self.t_step) :: int(self.rescale_time/self.t_step) , :]
        for node in range(self.num_nodes):
            signal = xpoints_new[::, node]
            signal = sp.signal_detrend(signal)
            
            if self.filter_band != None:
            
                sos = butter(2, [self.filter_band[0], self.filter_band[1]], 'bandpass', output='sos')
                filtered = sosfilt(sos, signal)
                
                signal_2 = filtered
            else:
                signal_2 = signal
            

            xpoints_new[::, node] = signal_2
            #signal without filtering to calculate the FCD
            xpoints_new_no_filter[::, node] = signal
        self.t_series = xpoints_new_no_filter
        
        self.FC = FC_cal(xpoints_new)

        phase = phases(xpoints_new)
        self.kuramoto = Kuramoto(phase)
        self.Metastability = np.std(self.kuramoto)
        self.Synchronization = np.average(self.kuramoto)

        
        t2 = process_time() 
        self.FCD = FCD_cal(xpoints_new_no_filter, self.time_window_duration, self.time_window_overlap, self.tstep_of_data)

        t3 = process_time() 
        print("Time elapsed in FCD calculation ", t3 - t2) # CPU seconds elapsed (floating point)
 
        print('The simulation has finished')
