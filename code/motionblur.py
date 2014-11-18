# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:04:41 2014

"""

from scipy.ndimage import rotate
from scipy.ndimage.filters import convolve
from scipy.misc import lena
from scipy.fftpack import fft2, fftshift
import numpy as np
#from numpy import zeros, ones
from math import floor, pow, sqrt

import matplotlib.pyplot as plt
from skimage.io import imshow, imread

#from skimage.io import imshow

def motionblur(size, angle):
    middle_row = floor(size / 2)
    psf = np.zeros([size, size])
    psf[middle_row, :] = np.ones([1, size])
    psf = rotate(psf, angle, reshape=False)
    return psf

def main():
    image = lena()
    #imshow(image)
    
    motionpsf = motionblur(15, 60)
    blurred = convolve(image, motionpsf)
    
    #Window the blurred image
    window = np.outer(np.hanning(image.shape[0]), np.ones(image.shape[0]))
    window = np.sqrt(window * window.T)
    
    blurred = np.multiply(window, blurred)
    fftblurred = fftshift(fft2(blurred))
    image2 = np.abs(fftblurred)
    image3 = image2/np.amax(image2)

    #imshow(blurred)
    plt.imshow(np.log(image3))
    plt.show()
    

    
if __name__ == "__main__":
    main()