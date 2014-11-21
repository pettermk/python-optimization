# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:27:40 2014

@author: Petter
"""

import numpy as np

from skimage.io import imshow, imread
from scipy.optimize import fmin, fmin_cg, fmin_bfgs, fmin_ncg, basinhopping

from math import sqrt, floor

def circlecost(x, *inputimage):
    #inputimage = np.asarray(inputimage[0]) if len(inputimage) == 1 else np.asarray(inputimage)
    inputimage = list(inputimage)
    size_x = len(inputimage)
    size_y = len(inputimage[0])
    x = [x[0]*size_x, x[1]*size_y, x[2]*max(size_x, size_y)]
    outersum = 0
    outercount = 0
    innersum = 0
    innercount = 0
    if x[0] < x[2] or x[0] > size_x- x[2]:
        return 1000
    if x[1] < x[2] or x[1] > size_y - x[2]:
        return 1000
    j_pow = []
    for k in range(floor(x[1] - x[2]), floor(x[1] + x[2]), 1):
        j_pow.append(pow(k - x[1], 2))
        
    for i in range(floor(x[0] - x[2]), floor(x[0] + x[2]), 1):
        if i >= size_x:
            continue
        i_squared = pow(i - x[0], 2)
        n = 0
        for j in range(floor(x[1] - x[2]), floor(x[1] + x[2]), 1):
            if j >= size_y:
                continue
            local_radius = sqrt(i_squared + j_pow[n])
            #if local_radius > x[2]:
            #    continue
            n +=1
            if local_radius > x[2]:
                outersum += inputimage[i][j]
                outercount +=1
            if local_radius <=  x[2]:
                innersum += inputimage[i][j]
                innercount += 1
    if outercount == 0 or innercount == 0:
        return 1000

    return -(innersum/innercount -outersum/outercount)
    
def circlecost_gradient(x, *inputimage):
    inputimage2 = np.asarray(inputimage[0]) if len(inputimage) == 1 else np.asarray(inputimage)
    deltax = 1 / inputimage2.shape[0]
    deltay = 1 / inputimage2.shape[1]
    deltar = 1 / max(inputimage2.shape)
    
    cost_x = circlecost(x, inputimage)
    df_x = circlecost(x + [deltax, 0, 0], inputimage) - cost_x
    df_y = circlecost(x + [0, deltay, 0], inputimage) - cost_x
    df_r = circlecost(x + [0, 0, deltar], inputimage) - cost_x
    
    return [deltax / df_x, deltay /df_y, deltar / df_r]
    
def drawcircle(inputimage, x, y, radius):
    x = floor(x*inputimage.shape[0])
    y = floor(y*inputimage.shape[1])
    radius = round(radius* max(inputimage.shape))
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
    x_opt = fmin(circlecost, [0.4, 0.7, 0.1], args=totuple(coins))        
    print(x_opt)
    coins2 = drawcircle(coins, x_opt[0], x_opt[1], x_opt[2])
    x_opt = fmin(circlecost, [0.2, 0.2, 0.1], args=totuple(coins2)) 
    coins3 = drawcircle(coins2, x_opt[0], x_opt[1], x_opt[2])
    x_opt = fmin(circlecost, [0.3, 0.8, 0.1], args=totuple(coins3)) 
    coins4 = drawcircle(coins3, x_opt[0], x_opt[1], x_opt[2])
    x_opt = fmin(circlecost, [0.8, 0.8, 0.1], args=totuple(coins4)) 
    coins5 = drawcircle(coins4, x_opt[0], x_opt[1], x_opt[2])
    x_opt = fmin(circlecost, [0.7, 0.4, 0.1], args=totuple(coins5)) 
    coins6 = drawcircle(coins5, x_opt[0], x_opt[1], x_opt[2])
    x_opt = fmin(circlecost, [0.7, 0.4, 0.1], args=totuple(coins6)) 
    coins7 = drawcircle(coins6, x_opt[0], x_opt[1], x_opt[2])
    #x_opt = fmin(circlecost, [0.7, 0.4, 0.1], args=totuple(coins7)) 
    #coins8 = drawcircle(coins7, x_opt[0], x_opt[1], x_opt[2])
    minimizer_kwargs = {'method' : 'Nelder-Mead', 'args' : totuple(coins7)}
    result = basinhopping(circlecost, [0.4, 0.7, 0.1], minimizer_kwargs=minimizer_kwargs)
    x_opt = result.x
    coins8 = drawcircle(coins7, x_opt[0], x_opt[1], x_opt[2])
    imshow(coins8)
        
if __name__ == "__main__":
    main()
        