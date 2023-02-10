import jax.numpy as jnp 
from jax import grad 
from jax import jacobian, vjp


def sigmoid(x): 
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

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
    return jnp.where(X > jnp.zeros(X.shape), X, np.zeros(X.shape))

def LRELU(X): 
    delta = 10e-4 
    return jnp.where(X > jnp.zeros(X.shape), X, delta * X) 


def tanh(X): 
    # return jnp.sinh(X)/jnp.cosh(X)
    return (jnp.exp(X) - jnp.exp(-X)) / (jnp.exp(X) + jnp.exp(-X))


def derivative(func):
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

    if func.__name__ != "tanh" and func.__name__ != 'softmax': 
        
        return grad(func)
    
    else: 
        # In the end, only the diagonal is used
        return jacobian(func)
