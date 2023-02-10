import jax.numpy as jnp 
from jax import grad 
from jax import jacobian, vjp
import numpy as np
import jax 


def sigmoid(x): 
    return 1.0 / (1.0 + jnp.exp(-x))

def CostLogReg(target): 

    def func(X):
        return -(1.0 / target.shape[0] * jnp.sum(
            (target * jnp.log(X + 10e-10)) + (1 - target * jnp.log(1 - X + 10e-10))))

    return func


def CostOLS(target): 

    def func(X): 
        return (1.0 / target.shape[0]) * jnp.sum((target - X) ** 2)

    return func


def CostCrossEntropy(target): 

    def func(X): 
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

def RELU(X): 
    return jnp.where(X > 0, X, 0)

def LRELU(X): 
    delta = 10e-4 
    return jnp.where(X > 0, 1 ,delta) 


def tanh(X): 
    # return jnp.sinh(X)/jnp.cosh(X)
    return (jnp.exp(X) - jnp.exp(-X)) / (jnp.exp(X) + jnp.exp(-X))


def derivate(func):
    """
    This is a wrapper class which uses autodiff library Jax for 
    computing the derivatives of activation functions. Unfortune-
    tly the grad() function included in Jax is not compatible 
    with the softmax or tanh functions, hence why the jacobian 
    of both functions is computed. Jax docs mention that vjp 
    function, which is the reverse mode vector-Jacobian product 
    of the function given as input, can be used for a slightly more
    efficient computation. 
    """

    return jacobian(func)
        # Since in all the cases encountered, the values we're 
        # looking for are contained in the diagonal of the 
        # Jacobian matrix, its possible to use the vjp function
        # in the following way: vjp(func, x) where x = input



# inputs = jnp.array(np.random.rand(8))
# targets = jnp.array([1])

# print(sigmoid(inputs))
# print(jax.nn.sigmoid(inputs))

# print(jnp.diagonal(jax.jacobian(jax.nn.sigmoid)(inputs)))
# print(jax.jacrev(sigmoid)(inputs))

# print(jax.nn.sigmoid(inputs)*(1 - jax.nn.sigmoid(inputs)))
# print(RELU(inputs))
#
# print(jnp.diagonal(jacobian(RELU)(inputs)))
# print(jnp.where(RELU(inputs) > 0, 1., 0.))
# print(vjp(RELU, inputs)[0])
# print('\n')
# print(jnp.diagonal(jacobian(LRELU)(inputs)))
# print(jnp.where(LRELU(inputs) > 0, 1., 0.))
# print(vjp(LRELU, inputs)[0])
