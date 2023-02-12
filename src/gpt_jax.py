import os 
os.environ['JAX_ENABLE_XLA'] = '1'
import jax.numpy as np
from jax import grad, jit, vjp, jacobian, jvp,jacfwd,jacrev, vmap
import time
import jax
import numpy
from autograd import grad, elementwise_grad
import autograd.numpy as anp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lrelu(x, alpha=0.01):
    return np.where(x >= 0, x, x *alpha) 


def matrix_sigmoid(x):
    return np.sum(sigmoid(x))

sigmoid = jit(sigmoid)

x = np.array(numpy.random.random((40,100)))

def sigmoid(x):
    return np.sum(1 / (1 + np.exp(-x)))

start = time.time()
der_sig = jax.grad(jit(sigmoid))(x)   
print(der_sig)
print(f'Time matrix grad: {time.time() - start}')



# Pure numpy implementation

def sigmoid(x): 
    return 1 / (1 + numpy.exp(-x))

x = numpy.random.random((40,100))

start = time.time() 
anal_der = sigmoid(x) * (1- sigmoid(x))
print(f'Time taken for analytical derivative:  {time.time() - start}')
print(anal_der)

x = anp.array(x)

from numba import njit

start = time.time() 
def sigmoid(x): 
    return 1 / (1 + anp.exp(-x))

grad_sig = elementwise_grad(sigmoid)(x)
print(grad_sig)
print(f'Time taken for grad() derivative:  {time.time() - start}')
