from scipy import fft

from pygamry.utils import nearest_index


# Fourier filter
# --------------
def fourier_band_filter(x, dt, f_min, f_max):
    # FFT
    x_fft = fft.rfft(x)
    f_fft = fft.rfftfreq(len(x), d=dt)

    # Set coef in band to zero
    start = nearest_index(f_fft, f_min)
    end = nearest_index(f_fft, f_max)
    x_fft[start:end] = 0
    
    # Invert FFT
    return fft.irfft(x_fft)
