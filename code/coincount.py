# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:27:40 2014

@author: Petter
"""

import time

from find_coin import cost_function
import numpy as np
from imageio import imread
from matplotlib.pyplot import imshow
from scipy.optimize import fmin, fmin_cg, fmin_bfgs, differential_evolution, basinhopping, minimize
from numba import jit, int32, int8, float64

from math import sqrt, floor


def circlecost(x, inputimage):
    #result = circlecost_helper(x, inputimage)
    # print(type(inputimage[0][0]))
    # print(inputimage.ndim)
    # return cost_function(x, inputimage)
    innervalue, outervalue = circlecost_helper(x, inputimage)
    if innervalue==0:
        return 2
    #return -(innervalue - outervalue)
    return outervalue / innervalue
    
def circlecost_outermean(x, inputimage):
    #innermean, outermean = circlecost_helper(x, inputimage)
    return outermean    

#@jit(float64(int32[:], int8[:,:]))
@jit
def circlecost_helper(x, inputimage):
    size_x = len(inputimage)
    size_y = len(inputimage[0])
    x = [x[0]*size_x, x[1]*size_y, x[2]*max(size_x, size_y)]
    outersum = 0.
    outercount = 0.
    innersum = 0.
    innercount = 0.
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
                outersum += pow(inputimage[i][j],2)
                outercount +=1
            if local_radius <=  x[2]:
                innersum += pow(inputimage[i][j],2)
                innercount += 1
    #if outercount == 0 or innercount == 0:
    #    return 0, 1000
        
    return innersum/innercount, outersum/outercount
    
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
    
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
        
def removecoin(inputimage, startguess=[0.5, 0.5, 0.1], method='local'):
    if method == 'local':
        x_opt = fmin(circlecost, [0.4, 0.7, 0.1], args=(inputimage))
        print(x_opt)
        outermean = circlecost_outermean(x_opt, totuple(inputimage))
        outputimage = drawcircle(inputimage, x_opt[0], x_opt[1], x_opt[2], 0)

    if method == 'global':
        minimizer_kwargs = {'method' : 'Nelder-Mead', 'args' : totuple(inputimage)}
        result = basinhopping(circlecost, [0.4, 0.7, 0.1], 
                              minimizer_kwargs=minimizer_kwargs)
        x_opt = result.x
        print(x_opt)
        outermean = circlecost_outermean(x_opt, totuple(inputimage))
        outputimage = drawcircle(inputimage, x_opt[0], x_opt[1], x_opt[2], 0)
        
    imshow(outputimage)        
    return outputimage

def drawcircle(inputimage, x, y, radius, new_value=30):
    outputimage = inputimage.copy()
    x = floor(x*inputimage.shape[0])
    y = floor(y*inputimage.shape[1])
    radius = round(radius* max(inputimage.shape))
    xx, yy = np.mgrid[:inputimage.shape[0], :inputimage.shape[1] ]
    index = (xx-x)**2 + (yy-y)**2 <= radius**2
    outputimage[index]=new_value
    return outputimage

def count_coins(threshold: float, max_no_coins: int = 100, optimizer: str = 'fmin', costfunc=circlecost):
    coins = imread('Images/coins.png')
    optimizers = {
        'diff_evo': lambda x: differential_evolution(x, bounds=[(0,1), (0,1), (0.01,0.2)], args=(coins,), seed=1),
        'fmin': lambda x: minimize(x, [0.4, 0.7, 0.1], method='Nelder-Mead', args=(coins,))
    }
    cost = 0
    found_coins = -1
    while cost < threshold and found_coins < max_no_coins:
        x_opt = optimizers[optimizer](circlecost)
        print(x_opt)
        coins = drawcircle(coins, *(x_opt.x))
        cost = x_opt.fun
        found_coins = found_coins + 1
    print('Found {} coins'.format(found_coins))
    imshow(coins)
        
if __name__ == "__main__":
    import sys
    t = time.process_time()
    count_coins(threshold=0.1, max_no_coins=100, optimizer='diff_evo')
    elapsed_time = time.process_time() - t
    print("Took %s seconds" % elapsed_time)

        