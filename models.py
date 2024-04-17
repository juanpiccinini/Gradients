import numpy as np
from time import process_time
import time
import numba as nb
import dask.array as da




def normalForm(degree, **kwargs):
    pass

def rossler_brain(R, t,  M, G, frequencies, constant, a, b, c):
    
    F = np.zeros( shape = (3, len(M)))
    gamma = 0.3
    numNodes = len(M)
    omega = frequencies
    
    x, y, z = R[0,:], R[1,:], R[2,:]
    
 
    xz = x*z
    
    fx = -omega*y - z*gamma +  gamma*(G*(np.dot(M, x)  - constant*x))
    fy = omega*x + a*y*gamma 
    fz = gamma*(b + xz - z*c)
    
    F[0,:], F[1, :], F[2,:] = fx, fy, fz
        
    
    return F

def hopf_brain(R, t, M, G, frequencies, constant, a):
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    omega = frequencies
    
    ##noise was left out given that the noise is introduced
    ##in the euler-mayurama integrator

    #noise = beta*np.random.randn(1, numNodes)*(1 + 1j)

    x, y = R[0,:], R[1,:]
    
    z = x + 1j*y
    
    zz = z*np.conjugate(z)
    parentesis = (a + 1j*omega - zz)
    fz = z*parentesis + G*(np.dot(M, z)  -constant*z) # + noise

    fx = fz.real
    fy = fz.imag

    F[0,:], F[1,:] = fx, fy
    
    
    return F




@nb.jit(nopython=True)
def hopf_brain_faster(X, t, M,  G, frequencies, constant, a, F= None, kinetic = None):
    if F is None:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) 
    
    else:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) + F

    
    return fz




def hopf_brain_2(R, t, M, G, frequencies, constant, a, beta):
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    omega = frequencies
    
    ##noise was left out given that the noise is introduced
    ##in the euler-mayurama integrator

    noise = beta*np.random.randn(1, numNodes)*(1 + 1j)

    x, y = R[0,:], R[1,:]
    
    z = x + 1j*y
    
    zz = z*np.conjugate(z)
    parentesis = (a + noise +  1j*omega - zz)
    fz = z*parentesis + G*(np.dot(M, z)  -constant*z)  + noise

    fx = fz.real
    fy = fz.imag

    F[0,:], F[1,:] = fx, fy
    
    
    return F



def bogdanov_takens_extended(X, t,  M, w_00, w_10, w_01, \
                             w_11, w_20, w_02,  w_21, w_12, w_30,  w_03, G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    x, y = X[0,:], X[1,:]
    
    fx = w_00 + w_10*x + w_01*y +  w_11*x*y + w_20*np.square(x) + \
              w_02*np.square(y) + w_21*np.square(x)*y \
                 + w_12*x*np.square(y) +  w_30*np.power(x, 3) + w_03*np.power(y, 3) \
                       + G*(np.dot(M, x)  - K*x) 
    fy = x  + G*(np.dot(M, y)  - K*y) 
    
    F[0,:], F[1,:] = fx, fy
    
    return F


def Hopf_extended(X, t,  M, wx_00, wx_10, wx_01, wx_20, wx_11, wx_02, wx_30, wx_21, wx_12, wx_03, wx_40, wx_31,  wx_22, wx_13, wx_04, \
        
         wx_50, wx_41, wx_32, wx_23, wx_14, wx_05, wy_00, wy_10, wy_01, wy_20, wy_11, wy_02, wy_30, wy_21, wy_12, wy_03, wy_40, wy_31,  wy_22, wy_13, wy_04, \
        
         wy_50, wy_41, wy_32, wy_23, wy_14, wy_05, G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    x, y = X[0,:], X[1,:]
    
    fx = wx_00 + wx_10*x + wx_01*y + wx_20*x**2 +  wx_11*x*y + wx_02*y**2  +  \
        + wx_30*x**3 +  wx_21*(x**2)*y + wx_12*x*y**2 + wx_03*y**3 + \
               + wx_40*x**4 + wx_31*(x**3)*y + wx_22*(x**2)*(y**2) + wx_13*x*(y**3)  + wx_04*y**4 + \
         + wx_50*x**5+ wx_41*(x**4)*y + wx_32*(x**3)*(y**2) + wx_23*(x**2)*(y**3) + wx_14*x*(y**4) +  wx_05*y**5 + \
          +   G*(np.dot(M, x)  - K*x) 
   
    
    fy = wy_00 + wy_10*x + wy_01*y + wy_20*x**2 +  wy_11*x*y + wy_02*y**2 +  \
         + wy_30*x**3 +  wy_21*(x**2)*y + wy_12*x*y**2 + wy_03*y**3 + \
               + wy_40*x**4 + wy_31*(x**3)*y + wy_22*(x**2)*(y**2) + wy_13*x*(y**3)  + wy_04*y**4 + \
        + wy_50*x**5+ wy_41*(x**4)*y + wy_32*(x**3)*(y**2) + wy_23*(x**2)*(y**3) + wy_14*x*(y**4) +  wy_05*y**5 + \
               + G*(np.dot(M, y)  - K*y) 
    
    F[0,:], F[1,:] = fx, fy
    
    return F


def Hopf_extended_2(X, t,  M, wx_00, wx_10, wx_01, wx_20, wx_11, wx_02, wx_30, wx_21, wx_12, wx_03, wx_40, wx_31,  wx_22, wx_13, wx_04, \
        
         wx_50, wx_41, wx_32, wx_23, wx_14, wx_05, wy_00, wy_10, wy_01, wy_20, wy_11, wy_02, wy_30, wy_21, wy_12, wy_03, wy_40, wy_31,  wy_22, wy_13, wy_04, \
        
         wy_50, wy_41, wy_32, wy_23, wy_14, wy_05, G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    x, y = X[0,:], X[1,:]
    
    poly = np.array([x, y, x**2, x*y, y**2,  \
        x**3, (x**2)*y, x*y**2, y**3,  \
               x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), y**4, \
         x**5, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4), y**5])
    
        
    
        
    fx = wx_00 + wx_10*poly[0] + wx_01*poly[1] + wx_20*poly[2] + wx_11*poly[3] + wx_02*poly[4] + wx_30*poly[5] + wx_21*poly[6] + wx_12*poly[7] + wx_03*poly[8] + wx_40*poly[9] + wx_31*poly[10] + wx_22*poly[11] + wx_13*poly[12] + wx_04*poly[13] + wx_50*poly[14] + wx_41*poly[15] + wx_32*poly[16] + wx_23*poly[17] + wx_14*poly[18] + wx_05*poly[19] + G*(np.dot(M, x)  - K*x) 
    fy = wy_00 +  wy_10*poly[0] + wy_01*poly[1] + wy_20*poly[2] + wy_11*poly[3] + wy_02*poly[4] + wy_30*poly[5] + wy_21*poly[6] + wy_12*poly[7] + wy_03*poly[8] + wy_40*poly[9] + wy_31*poly[10] + wy_22*poly[11] + wy_13*poly[12] + wy_04*poly[13] + wy_50*poly[14] + wy_41*poly[15] + wy_32*poly[16] + wy_23*poly[17] + wy_14*poly[18] + wy_05*poly[19]  +  G*(np.dot(M, y)  - K*y) 

    
    F[0,:], F[1,:] = fx, fy
    
    return F




@nb.jit(nopython=True)
def hopf_brain_ephaptic(X, t, M, Length_inv,  G, G_ephap, frequencies, constant,constant_ephap, a):

    
    numNodes = len(M)
    omega = frequencies

    z = X[0]
    zz = z*np.conjugate(z)
    parentesis = (a + 1j*omega - zz)

    
    #without delay
    fz = (z*parentesis + G*(M.dot(z)  -constant*z) 
          
            +  G_ephap*(Length_inv.dot(z)  -  constant_ephap*z))

    # fz = (z*parentesis + G*(np.dot(M, z)  -constant*z)
          
    #       +  G_ephap*np.dot(Length_inv, z)*z)
          
    return fz




def hopf_brain_ephaptic_2(t, y, a, frequencies, M, Length_inv, constant,constant_ephap,  G, G_ephap, noise_amp):
    
    numNodes = len(M)
    F = np.empty(numNodes, dtype = np.complex64)
    omega = frequencies

    z = y
    zz = z*np.conjugate(z)
    parentesis = (a + 1j*omega - zz)
    dW = (np.random.randn(numNodes) + 1j*np.random.randn(numNodes))

    

    fz = (z*parentesis + G*(np.dot(M, z)  -constant*z) 
          
            +  G_ephap*(np.dot(Length_inv, z)  -  constant_ephap*z))
    
    
    F = fz + noise_amp*dW

    
    return F

def hopf_brain_delay(X, t, M, Length_inv, Delays,  G, G_ephap, frequencies, constant,constant_ephap, a):

    F = np.zeros_like(X, dtype = np.complex128)
    omega = frequencies
    
    z = X[0, :]
    


    
    z_delay = Z_delayed_computatation(X, Delays)

    zz = z*np.conjugate(z)
    parentesis = (a + 1j*omega - zz)

    
    # fz = (z*parentesis + G*(np.sum(M*z_delay, axis = 1)  -constant*z) 
          
    #       +  G_ephap*(np.dot(Length_inv, z)  -  constant_ephap*z))    
    
    # fz = (z*parentesis + G*(np.sum(M*z_delay, axis = 1)  -constant*z) 
          
    #       +  G_ephap*(np.dot(Length_inv, z))*z)  
    
    #without delay
    fz = (z*parentesis + G*(np.dot(M, z)  -constant*z) 
          
            +  G_ephap*(np.dot(Length_inv, z)  -  constant_ephap*z))
    
    
    F[1:,:] = X[0:-1,:]
    F[0,:] = fz
    
    return F






@nb.jit(nopython=True)
def Z_delayed_computatation(X, Delays):
  
  numNodes = Delays.shape[0]
  z_delay = np.zeros((numNodes, numNodes), dtype = complex128)
  
  for node in range(numNodes):
    Delay_node = Delays[node, :]
    z_delay[:, node] = X[Delay_node, node]
  
  
  return z_delay


def new_model(X, t,  M, frequencies, w0, w1, w2, w3, w4, w5, a,  G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    rho, theta = X[0,:], X[1,:]
    
    fx = w0 + w1*rho + w2*rho**2 + w3*rho**3 + w4*rho**4 + w5*rho**5 
    fy = frequencies + a*rho + G*(np.dot(M, theta)  - K*theta) 
    
    F[0,:], F[1,:] = fx, fy
    
    return F


def new_model_3d(X, t,  M, wx_000 , wx_100 , wx_010, wx_001, wx_200, wx_110, wx_101, wx_020, wx_011,  wx_002, wx_300, wx_210, wx_120, wx_030, wx_201, wx_102, wx_021, wx_012, wx_003, wy_000 , wy_100 , wy_010, wy_001, wy_200, wy_110, wy_101, wy_020, wy_011, wy_002, wy_300, wy_210, wy_120, wy_030, wy_201, wy_102, wy_021, wy_012, wy_003, wz_000 , wz_100 , wz_010, wz_001, wz_200, wz_110, wz_101, wz_020, wz_011, wz_002, wz_300, wz_210, wz_120, wz_030, wz_201, wz_102, wz_021, wz_012, wz_003, G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (3, numNodes))
    x, y, z = X[0,:], X[1,:], X[2,:]
    
    poly = np.array([x, y, z, x**2, x*y, x*z, y**2, y*z, z**2, \
    x**3, (x**2)*y, x*y**2, y**3, (x**2)*z, x*z**2, (z**2)*y, z*y**2,  z**3])
    
        
    
        
    fx = wx_000 + wx_100*poly[0] + wx_010*poly[1] + wx_001*poly[2] + wx_200*poly[3] + wx_110*poly[4] + wx_101*poly[5] + wx_020*poly[6] + wx_011*poly[7] + wx_002*poly[8] + wx_300*poly[9] + wx_210*poly[10] + wx_120*poly[11] + wx_030*poly[12] + wx_201*poly[13] + wx_102*poly[14] + wx_021*poly[15] + wx_012*poly[16] + wx_003*poly[17] + G*(np.dot(M, x)  - K*x) 
    fy = wy_000 + wy_100*poly[0] + wy_010*poly[1] + wy_001*poly[2] + wy_200*poly[3] + wy_110*poly[4] + wy_101*poly[5] + wy_020*poly[6] + wy_011*poly[7] + wy_002*poly[8] + wy_300*poly[9] + wy_210*poly[10] + wy_120*poly[11] + wy_030*poly[12] + wy_201*poly[13] + wy_102*poly[14] + wy_021*poly[15] + wy_012*poly[16] + wy_003*poly[17]  +  G*(np.dot(M, y)  - K*y) 
    fz = wz_000 + wz_100*poly[0] + wz_010*poly[1] + wz_001*poly[2] + wz_200*poly[3] + wz_110*poly[4] + wz_101*poly[5] + wz_020*poly[6] + wz_011*poly[7] + wz_002*poly[8] + wz_300*poly[9] + wz_210*poly[10] + wz_120*poly[11] + wz_030*poly[12] + wz_201*poly[13] + wz_102*poly[14] + wz_021*poly[15] + wz_012*poly[16] + wz_003*poly[17] +  G*(np.dot(M, z)  - K*z) 
    
    F[0,:], F[1,:], F[2,:] = fx, fy, fz
    
    return F
