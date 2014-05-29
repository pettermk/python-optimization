# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:04:41 2014

"""

from scipy.ndimage import rotate
from scipy.ndimage.filters import convolve
from scipy.misc import lena
from numpy import zeros, ones
from math import floor

from skimage.io import imshow

def motionblur(size, angle):
    middle_row = floor(size / 2)
    psf = zeros([size, size])
    psf[middle_row, :] = ones([1, size])
    psf = rotate(psf, angle, reshape=False)
    return psf

def main():
    image = lena()
    imshow(image)
    
    motionpsf = motionblur(50, 45)
    blurred = convolve(image, motionpsf)
    imshow(blurred)
    
if __name__ == "__main__":
    main()