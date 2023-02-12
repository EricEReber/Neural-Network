import jax.numpy as np
from jax import grad, jit, vjp, jacobian, jvp,jacfwd,jacrev, vmap
import time
import jax
import numpy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lrelu(x, alpha=0.01):
    return np.where(x >= 0, x, x *alpha) 


def matrix_sigmoid(k, x):
    return np.sum(sigmoid(x))

sigmoid = jit(sigmoid)

# calculate the derivative of the matrix sigmoid function
matrix_sigmoid_derivative = grad(matrix_sigmoid)

# test the derivative of the matrix sigmoid function

x = np.array(numpy.random.random((40,100)))

start = time.time()
analytical_derivative = sigmoid(x) * (1 - sigmoid(x))
print(f'Time taken for analytical derivative:  {time.time() - start}')
print(analytical_derivative)


# Pure numpy implementation

def sigmoid(x): 
    return 1 / (1 + numpy.exp(-x))

x = numpy.random.random((40,100))

start = time.time() 
anal_der = sigmoid(x) * (1- sigmoid(x))
print(f'Time taken for analytical derivative:  {time.time() - start}')
print(anal_der)



# start = time.time()
# sig_der = vmap(jacfwd(jit(sigmoid)))(x)  
#
# print(sig_der.shape)
# functional = np.diag(sig_der[:,:,0])
# print(functional) 
# print(f'Time taken for jax autodiff solution : {time.time() - start}')
