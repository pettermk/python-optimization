# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:04:41 2014

"""

from scipy.ndimage import rotate
#from scipy.ndimage.filters import convolve
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from skimage.color import rgb2gray
from skimage.restoration import richardson_lucy, unsupervised_wiener
from skimage.data import lena
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
    psf = psf/sum(psf)
    return psf
    
def blurimage(image, length, angle, plot=True):
    motionpsf = motionblur(length, angle)
    blurred = convolve2d(image, motionpsf, 'same')
    blurred += 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    plt.imshow(blurred)
    return blurred
    
def deblurimage(image, length, angle, plot=True):
    motionpsf = motionblur(length, angle)
    deblurred, _ = unsupervised_wiener(image, motionpsf)
    plt.imshow(deblurred, vmin=deblurred.min(), vmax=deblurred.max())
    return deblurred

def fftdeblurimage(image, length, angle, plot=True):
    psf = motionblur(length, angle)
    diff = len(image) - len(psf)
    xpad = (diff // 2)
    ypad = xpad + diff % 2
    psf = np.pad(psf,((xpad, ypad), (xpad, ypad)), 'constant', constant_values=(0,0))
    fftpsf = fft2(psf)
    fftinput = fft2(image)
    fftresult = fftinput / fftpsf
    result = fftshift(abs(ifft2(fftresult)))
    if plot == True:
        plt.imshow(result)
    
    return result
    
def fftblurimage(image, length, angle, plot=True):
    psf = motionblur(length, angle)
    diff = len(image) - len(psf)
    xpad = (diff // 2)
    ypad = xpad + diff % 2
    psf = np.pad(psf,((xpad, ypad), (xpad, ypad)), 'constant', constant_values=(0,0) )
    fftpsf = fft2(psf)
    fftinput = fft2(image)
    fftresult = fftinput * fftpsf
    result = fftshift(abs(ifft2(fftresult)))
    if plot == True:
        plt.imshow(result)
    
    return result

def main():
    image = lena()
    #imshow(image)
    
    """motionpsf = motionblur(15, 60)
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
    plt.show()"""
    

    
if __name__ == "__main__":
    main()