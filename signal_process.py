import numpy as np
from scipy.fftpack import fftfreq, fft, ifft, fft2
import matplotlib.pyplot as plt


def signal_detrend(t_series):
    """It detrends the signal by subtracting the mean and
    then normalazing it divinding by the standar deviation

    The input should be an (1 x time) array"""
    
    mean = np.mean(t_series)
    std = np.std(t_series)
    signal = (t_series - mean)/std


    return signal




def freq_filter(t_series, time_step, freq_min, freq_max):
    """Filter the signal only in the bandwidth selected"""

    num_points = t_series.size

    freq = fftfreq(num_points, time_step)
    mask = freq >= 0
    fft_values = fft(t_series)
    fft_theo = 2*np.abs(fft_values/num_points)
    fft_theo = fft_theo[mask]

    cut_fft_values = fft_values.copy()
    cut_fft_values[np.abs(freq) > freq_max] = 0 # cut signal above freq_max
    cut_fft_values[np.abs(freq) < freq_min] = 0 # cut signal below freq_min

    new_signal = ifft(cut_fft_values)
    TS_new = np.real(new_signal)

    return TS_new

def fft_2d(matrix):
    # Take the fourier transform of the image.
    F1 = np.fft.fft2(matrix)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = np.fft.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )

    return psd2D

def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.log10(im_fft + 1))
    plt.colorbar()
    plt.title('2D Fourier transform')
  
