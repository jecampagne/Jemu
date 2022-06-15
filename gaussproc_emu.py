

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
jax.config.update("jax_enable_x64", True)

from transformation import *
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
def square_scaled_distance(X, Z,lengthscale = 1.):
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
def kernel_DOT(X: jnp.ndarray,
               Z: jnp.ndarray,
               C: jnp.ndarray)->jnp.ndarray:
    """
    DOT product kernel
    """
    return X @ C @ Z.T


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

@jit
def kernel_Matern32(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=3/2 kernel
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt3_r = 3**0.5 * r
    k = params["k_scale"] * (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

@jit
def kernel_Matern52(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=5/2 kernel
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = params["k_scale"] * (1.0 + sqrt5_r + sqrt5_r**2 /3.0) * jnp.exp(-sqrt5_r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

# ########## Helpers ##########

@partial(jit, static_argnums=(0,))  #kernel is a function,   
def _simple_predict(kernel,
                    x_star,
                    phi_star,
                    x_train,
                    kernel_hat,
                    beta_hat,
                    kinv_XX_res, 
                    mean_function
                    ):
        """
        theta_star: a vector of new theta values (N*,d)
        
        Make prediction after optimization
        y_\ast = \Phi_\ast \hat{\beta} 
                + \hat{k_\ast}^T \hat{K_y}^{-1} (y_train - \Phi(X_train)\hat{\beta})        
        """
        
        k_pX = kernel(x_star, x_train, kernel_hat, noise=0.0, jitter=0.0)#jitter=self.jitter)

        #k_pX (N*,N) the transpose not to be as it is k(X*,Xtr) (and not k(Xtr,X*) as in RW book)
        y_star = phi_star @ beta_hat + k_pX @ kinv_XX_res  + mean_function

        return y_star.squeeze()

# ########## Class #############
class GPEmuBase():
    def __init__(self, 
                 kernel: Callable[[jnp.ndarray, 
                              jnp.ndarray, 
                              Dict[str, jnp.ndarray], 
                              float, float],jnp.ndarray] = kernel_RBF, 
                 order: int = 2,
                 x_trans: bool = True,
                 y_trans: bool = False,
                 use_mean: bool = False) -> None:
        
        # order of the poynomial regression
        # we support only second order here
        self.order = order
        if self.order > 2:
            msg = 'At the moment, we support only order = 1 and order = 2'
            raise RuntimeError(msg)
        

        # Gaussian Process kernel
        self.kernel = kernel  
        
        # choose to make transformation
        self.x_trans = x_trans
        self.y_trans = y_trans

        # if we want to centre the output on zero
        self.use_mean = use_mean

    
    def compute_basis(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the input basis functions 
        .. math::
                \Phi = [1, \theta_1,\dots,\theta_p, \theta_1^2,\dots,\theta_p^2, \dots, \theta_p^d]
        """

        dummy_phi = [X**i for i in jnp.arange(1, self.order + 1)]
        phi = jnp.concatenate(dummy_phi, axis=1)
        phi = jnp.c_[np.ones((X.shape[0], 1)), phi]
        return phi


class GPEmu(GPEmuBase):
    def __init__(self,
            kernel: Callable[[jnp.ndarray, 
                              jnp.ndarray, 
                              Dict[str, jnp.ndarray], 
                              float, float],jnp.ndarray] = kernel_RBF,
            order: int = 2,
            x_trans: bool = True,
            y_trans: bool = False,
            use_mean: bool = False) -> None:
        
        super().__init__(kernel,order,x_trans,y_trans,use_mean)

        
    def load_info(self, folder_name: str, file_name: str)->None:
        """
        load info for prediction only
        """
        
        fname = folder_name + '/' + file_name + '.npz'
        
        data = np.load(fname, allow_pickle=True)
        self.x_train    = data["x_train"]
        ###self.y_train    = data["y_train"]   #not really necessary
        self.mean_theta = data['mean_theta']
        self.x_trans    = data['x_trans'][0]
        self.y_trans    = data['y_trans'][0]
        self.kernel_hat = data["kernel_hat"].item()
       
        self.beta_hat      = data["beta_hat"]
        self.kinv_XX_res   = data["kinv_XX_res"]
        self.mean_function = data["mean_function"]
        mu_matrix       = data['mu_matrix']
        self.transform  = transformation(
            jnp.zeros(shape=(self.beta_hat.shape[0]+1,
                             self.beta_hat.shape[0])),
                            jnp.zeros(shape=(self.beta_hat.shape[0]+1,1))) #fake

        self.transform.mu_matrix = mu_matrix
        self.transform.y_min = data['y_min'][0]

    
    def simple_predict(self, theta_star: jnp.ndarray) -> jnp.ndarray:
        """
        theta_star: a vector of new theta values (N*,d)
        
        Make prediction after optimization
        y_\ast = \Phi_\ast \hat{\beta} 
                + \hat{k_\ast}^T \hat{K_y}^{-1} (y_train - \Phi(X_train)\hat{\beta})        
        """

        if self.kinv_XX_res is None:
            msg = 'kinv_XX_res is not set...'
            raise RuntimeError(msg)

        x_star = theta_star - self.mean_theta  # shift computed at training phase  (mean_theta is a mtx)
        if self.x_trans: # use the whitening transform computed at training phase
            x_star = self.transform.x_transform_test(x_star)        
        
        phi_star = self.compute_basis(x_star)   # (N* x m)
        
        #jitted helper
        return _simple_predict(self.kernel,
                               x_star,
                               phi_star,
                               self.x_train,
                               self.kernel_hat,
                               self.beta_hat,
                               self.kinv_XX_res, 
                               self.mean_function)

    
    def pred_original_function(self, theta_star: jnp.ndarray) -> jnp.ndarray:
        '''
        Calculates the original function if the log_10 transformation is used on the target.
        :param: theta_star (np.ndarray) - the test point in parameter space

        :return: y_original (np.ndarray) - the predicted function in the linear scale (original space) is returned
        '''

        if not self.y_trans:
            msg = 'You must transform the target in order to use this function'
            raise RuntimeWarning(msg)

            
        

        mu = self.simple_predict(theta_star)
        y_original = self.transform.y_inv_transform_test(mu)
        return y_original

    
    def predict(self, theta_star: jnp.ndarray) -> jnp.ndarray:
        """
        prediction according to y_trans
        """
        if self.y_trans:
            return self.pred_original_function(theta_star)
        else:
            return self.simple_predict(theta_star)
        
            
    
