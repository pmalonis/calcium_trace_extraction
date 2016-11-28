 from __future__ import division
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
#from scipy.stats import multivariate_normal

def get_hp_kernel(r, sigma):
    '''Get 2-d kernel used to high-pass filter raw data
    Parameters:
    
    r: average neuron size
    sigma: 
    '''

    h_func = lambda x,y: np.exp(-(x**2+y**2)/(2*sigma**2))
    omega_range = np.arange(-r,r+1)
    omega_x, omega_y = np.meshgrid(omega_range,omega_range)
    h = h_func(omega_x, omega_y)
    h = h - np.sum(h)/h.size

    return h

def simulate_ca_trace(rate, T, sr=20, tau=0.5):
    '''Simulates a ca trace using an AR(1) model
    rate: firing rate of neuron in Hz
    T: length of time to simulate
    sr: sampling rate
    tau: decay time constant in seconds
    '''

    n = T*sr
    s = np.random.poisson(rate/sr, size=n)  #spikes

    gamma = np.exp(-1/(sr*tau))
    c = np.zeros(n)  #ca trace
    for i in xrange(1, n):
        c[i] = gamma*c[i-1] + s[i]

    return c,s

def create_footprints(n, x_max, y_max, r):
    '''Creates spatial footprint matrix of cells
    
    Parameters:
    n: number of cells
    w: width of field of view
    h: height of field of view 
    r: cell radius'''

    sigma = .75*r
    h_func = lambda x,y: np.exp(-(x**2+y**2)/(2*sigma**2))
    omega_range = np.arange(-r,r+1)
    omega_x, omega_y = np.meshgrid(omega_range,omega_range)
    h = h_func(omega_x, omega_y)
    h[omega_x**2+omega_y**2 >r**2] = 0

    A = np.zeros((x_max*y_max, n))
    fovs = np.zeros((x_max,y_max,n))
    for i in xrange(n):
        fov = np.zeros((x_max, y_max))
        x = np.random.randint(r, x_max-r-1)
        y = np.random.randint(r, y_max-r-1)
#        import pdb;pdb.set_trace()
        fov[y-r:y+r+1, x-r:x+r+1] = h
        A[:,i] = fov.reshape(x_max*y_max)
        fovs[:,:,i] = fov

    return A,np.sum(fovs,2)

def create_background(T, x_max, y_max, sr=20):
    noise = np.random.randn(T, x_max, y_max)
    #10 filter window is 10s and half fov in each dimension
    sigma = [sr*10, x_max/2, y_max/2]  
    filtered = gaussian_filter(noise, sigma)
    frames = filtered + np.random.randn(*filtered.shape)*.05
    
    return frames

def write_video(fname, frames, sr=20):
    fourcc=cv2.cv.CV_FOURCC(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc, sr,frames.shape[1:],False)
    for fr in frames:
        out.write(fr)

    out.release()

    return
    
