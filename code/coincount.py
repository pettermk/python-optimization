# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:27:40 2014

@author: Petter
"""

import numpy as np

from skimage.io import imshow, imread
from scipy.optimize import fmin, fmin_cg, fmin_bfgs

from math import sqrt, floor

def helper_circlecost(x):
    coins = imread('Images/coins.png')
    x = [x[0]*coins.shape[0], x[1]*coins.shape[1], x[2]*max(coins.shape)]
    return circlecost(x, coins)

def circlecost(x, inputimage):
    outersum = 0
    outercount = 0
    innersum = 0
    innercount = 0
    if x[0] < x[2] or x[0] > inputimage.shape[0] - x[2]:
        return 1000
    if x[1] < x[2] or x[1] > inputimage.shape[1] - x[2]:
        return 1000

    for i in range(floor(x[0] - x[2]), floor(x[0] + x[2]), 1):
        for j in range(floor(x[1] - x[2]), floor(x[1] + x[2]), 1):
            local_radius = sqrt(pow(i - x[0], 2) + pow(j - x[1], 2))
            if local_radius > 2 * x[2]:
                continue
            if local_radius > x[2]:
                outersum += pow(inputimage[i, j], 2)
                outercount +=1
            if local_radius <=  x[2]:
                innersum += pow(inputimage[i, j], 2)
                innercount += 1
    if outercount == 0 or innercount == 0:
        return 1000

    return (outersum/outercount) / (innersum/innercount)
    
def drawcircle(inputimage, x, y, radius):
    x = floor(x*inputimage.shape[0])
    y = floor(y*inputimage.shape[1])
    radius = floor(radius* max(inputimage.shape))
    xx, yy = np.mgrid[:inputimage.shape[0], :inputimage.shape[1] ]
    index = (xx-x)**2 + (yy-y)**2 <= radius**2
    inputimage[index]=0
    return inputimage

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def main():
    coins = imread('Images/coins.png')
    x_opt = fmin(helper_circlecost, [0.7, 0.3, 0.1])  
    print(x_opt)
    coins2 = drawcircle(coins, x_opt[0], x_opt[1], x_opt[2])
    imshow(coins2)
        
if __name__ == "__main__":
    main()
        