

import numpy as np 
# for I/O
import os          

from functools import partial

from numbers import Number

from typing import Union, Dict, Callable, Optional, Tuple
import jax
import jaxopt
import jax.numpy as jnp
import jax.scipy as jsc
from jax import vmap, jit

from jax.tree_util import register_pytree_node_class, Partial       ##### New approach: Custom PyTree

jax.config.update("jax_enable_x64", True)

#from transformation import *
from helper import *

"""
        y(x) = \Phi(x).\beta + GP(0,K(x,x^\prime; \alpha)
        \alpha = (amplitude, \{\ell \}_i<d)  with d dimension of x
        K: GP kernel as RBF

        Determination of \beta & \alpha based on: 
        Gaussian process class
        Using C. E. Rasmussen & C. K. I. Williams, 
        Gaussian Processes for Machine Learning, the MIT Press, 2006 

        2.7 Incorporating Explicit Basis Functions
        Eqs. 2.39-2.41 (mean only)
        Eq. 2.43 Marginal Likelihood
        Hypothesis: beta prior=N(0,\lambda_cap Id) => simplififications
        
        
        Nb.
        jax.lax.linalg.cholesky(k_XX) equals jsc.linalg.cholesky(k_XX, lower=True)
                                            
        jax.lax.linalg.triangular_solve(chol_XX, y_train[:,jnp.newaxis], lower=True,left_side=True)
        equals
        jsc.linalg.solve_triangular(chol_XX, y_train[:,jnp.newaxis], lower=True)
        
"""



@jit
def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)

@jit
def square_scaled_distance(X, Z,lengthscale):
    """
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)

########### Kernels #############

@jit
def kernel_RBF(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    RBF kernel
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    k = params["k_scale"] * jnp.exp(-0.5 * r2)
    if X.shape == Z.shape:
        k +=  (noise + jitter) * jnp.eye(X.shape[0])
    return k

@jit
def kernel_Matern12(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=1/2 kernel; exponentiel decay
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    k = params["k_scale"] * jnp.exp(-r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

# ########## Helpers ##########


@jit
def _pred_original_function(gp, theta_star):
    '''
    Calculates the original function if the log_10 transformation is used on the target.
    :param: theta_star (np.ndarray) - the test point in parameter space

    :return: y_original (np.ndarray) - the predicted function in the linear scale (original space) is returned
    '''

    mu = _simple_predict(gp,theta_star)
    y_original = jnp.power(10, mu) + 2.0 * gp.y_min
    return y_original


@jit
def _simple_predict(gp, theta_star):
        """
        theta_star: a vector of new theta values (N*,d)
        
        Make prediction after optimization
        y_\ast = \Phi_\ast \hat{\beta} 
                + \hat{k_\ast}^T \hat{K_y}^{-1} (y_train - \Phi(X_train)\hat{\beta})        
        """


        x_star = theta_star - gp.mean_theta  # shift computed at training phase  (mean_theta is a mtx)

        #whitening
        x_star = jnp.atleast_2d(x_star)
        x_star = x_star @ gp.mu_matrix.T 

        #compute Phi(x_star) basis [1, x*_1, x*_2..., x*_5, (x*_1)^2, (x*_2),...,  (x*_5)^2]
        dummy_phi = vmap(vmap(lambda x,i:x**i, in_axes=(None, 0)),
             in_axes=(0, None))(x_star,jnp.linspace(1,gp.order,gp.order)).reshape(x_star.shape[0],-1)
        phi_star = jnp.c_[jnp.ones((x_star.shape[0], 1)), dummy_phi]

        
        ##k_pX = gp.kernel(x_star, gp.x_train, gp.kernel_hat, noise=0.0, jitter=0.0)
        k_pX = gp.kernel(X=x_star, Z=gp.x_train)

        #k_pX (N*,N) the transpose not to be as it is k(X*,Xtr) (and not k(Xtr,X*) as in RW book)
        y_star = phi_star @ gp.beta_hat + k_pX @ gp.kinv_XX_res  + gp.mean_function

        return y_star.squeeze()




# ########## Class #############
# JEC 2/7/22
# simplify code using the fact that
#  x_trans, #  bool = True,
#  y_trans, #: bool = True,

@register_pytree_node_class
class GPEmu:
    def __init__(self,
                 order, # : int
                 kernel, #: Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], float, float],jnp.ndarray]
                 x_train,  #: ndarray
                 mean_theta, # ndarray
                 ####kernel_hat, # dict
                 beta_hat,   # ndarray
                 kinv_XX_res,  # ndarray
                 mean_function,  # ndarray
                 mu_matrix,  # ndarray
                 y_min   # float
                 ):


        # Gaussian Process kernel
        self.kernel = kernel

        # order of the poynomial regression
        # we support only second order here
        self.order = order
        
        
        self.x_train = x_train
        self.mean_theta = mean_theta
        ####self.kernel_hat = kernel_hat
        self.beta_hat = beta_hat
        self.kinv_XX_res = kinv_XX_res
        self.mean_function = mean_function
        self.mu_matrix = mu_matrix
        self.y_min = y_min
        
        
    ###########
    ## NEW ####
    ###########
    predict =  _pred_original_function

    def tree_flatten(self):
        children = (self.kernel,
            self.x_train,       # ndarray
                    self.mean_theta,    # ndarray
                    self.beta_hat,      # ndarray
                    self.kinv_XX_res,   # ndarray
                    self.mean_function, # ndarray
                    self.mu_matrix,     # ndarray
                    self.y_min          # float
                    )
        aux_data = {'order': self.order}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        order = aux_data['order']
        return cls(order, *children)

    




    
        
            
    
