from random import random, seed
from numpy import array, array_equal

# Setup random numbers pure python
seed(1)
vec1 = [random() for i in range(1000000)]
vec2 = [random() for i in range(1000000)]

# Make equivalent numpy vectors
np_vec1 = array(vec1)
np_vec2 = array(vec2)

# Create results in both pure python and numpy
correct = [x[0] * x[1] for x in zip(vec1, vec2)]
np_correct = array(correct)

def elementwise_multiply_random():
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] * vec2[i])      
    return result
    
def elementwise_multiply_random_using_zip():
    return [x[0] * x[1] for x in zip(vec1, vec2)]

def elementwise_multiply_random_using_np():
    return np_vec1 * np_vec2


%timeit elementwise_multiply_random()
assert elementwise_multiply_random() == correct
    
%timeit elementwise_multiply_random_using_zip()
assert elementwise_multiply_random_using_zip() == correct

%timeit elementwise_multiply_random_using_np()
assert array_equal(elementwise_multiply_random_using_np(), np_correct)

