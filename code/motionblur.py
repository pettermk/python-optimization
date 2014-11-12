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
    
def circlecost(inputimage, x, y, radius):
    outersum = 0
    innersum = 0
    for i in range(x - radius, x + radius, 1):
        for j in range(y - radius, y + radius, 1):
            local_radius = sqrt(pow(i - x, 2) + pow(j - y, 2))
            if local_radius > 2 * radius:
                continue
            if local_radius > radius:
                outersum += pow(inputimage[i, j], 2)
            if local_radius <=  radius:
                innersum += pow(inputimage[i, j], 2)
    return outersum / innersum
    
if __name__ == "__main__":
    main()