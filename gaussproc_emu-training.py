

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

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
numpyro.util.enable_x64()

from transformation import *
from helper import *

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
@partial(jit, static_argnums=(1,))  #kernel is a function
def _marginal_likelihood(params, 
                         kernel, 
                         lambda_cap, phi, x_train, var, y_train, n_trains):
    """
    Concerte implementation of
    Marginal Likelihood with hypothesis of Gaussian Prior for beta
    
    Nb JEC if we want to flag kargs as a static_arg then we should prefer a namedtuple due to hashtable
    """
    print("_marginal compil...")
    
    
    # Extract informations
    k_scale =  params[0]
    k_length = params[1:]
    kernel_params= {"k_scale":k_scale, "k_length":k_length}


#    lambda_cap = kargs["lambda_cap"]
#    phi        = kargs["phi"]
#    #KGP        = kargs["KGP"]
#    x_train    = kargs["x_train"]
#    var        = kargs["var"]
#    y_train    = kargs["y_train"]
#    n_trains   = kargs["n_trains"]

    # beta prior = N(0, lambda_cap * Id)
    # basis_cov = lambda_cap \Phi(X_train) \Phi(X_train)^T  (NxN matrix)
    # basis_cov can be computed once but NxN matrix -> save storage = compute each time

    basis_cov = lambda_cap * (phi @ phi.T)

    # compute GP K(X,X)
    KGP = kernel(x_train,x_train, 
                        params=kernel_params,
                        noise=var, 
                        jitter=0.0)#self.jitter)

    
    # Kernel tot 
    K_XX = KGP + basis_cov

    # the beta prior mean is 0
    y_diff = y_train[:,jnp.newaxis]

    ### just to use LAX as default
    #chol_XX = jsc.linalg.cholesky(K_XX, lower=True)
    #v  = jsc.linalg.solve_triangular(chol_XX, y_diff, lower=True)

    chol_XX = jax.lax.linalg.cholesky(K_XX)
    v  = jax.lax.linalg.triangular_solve(chol_XX, y_diff, lower=True, left_side=True)


    mlp  = -0.5 * v.T @ v                       # chi2
    mlp += -jnp.sum(jnp.log(jnp.diag(chol_XX)))  # complexity term (determinent) 
                            # nb. 1/2Log(det(A)) = Tr[Log[Chol_A]]= sum_i Log[Chol_A_ii]
    mlp += -0.5 * n_trains * jnp.log(2.*jnp.pi)   # cte not really necessary in the min

    return mlp.squeeze()

@partial(jit, static_argnums=(0,1))  #kernel is a function, nbasis is used for identity
def _post_training(kernel,nbasis,
                  x_train,kernel_hat,var,jitter, y_train, phi, lambda_cap):
        
        # compute some quanties for predicton
        k_XX = kernel(x_train,x_train, params=kernel_hat, 
                         noise=var, jitter=jitter)  #aka Ky
                
        chol_XX = jax.lax.linalg.cholesky(k_XX)    # K_XX = Lxx Lxx^T
        
        # beta_hat = W_\beta^{-1} [\Phi^T K_y^{-1} y + C^{-1}\mu]      with \mu=0 C=\lambda_{cap} Id
        # W_\beta = C^{-1} + \Phi^T K_y^{-1} \Phi

        #kinv_XX_y = K_y^{-1} y   => kinv_XX_y=Lxx^T\(Lxx\y)  with  
        kinv_XX_y = jax.lax.linalg.triangular_solve(
            chol_XX.T, 
            jax.lax.linalg.triangular_solve(chol_XX, y_train[:,jnp.newaxis],
                                            lower=True,  left_side=True),
            lower=False,
            left_side=True)
        
        

        v1 = phi.T @ kinv_XX_y
        v2 = jax.lax.linalg.triangular_solve(chol_XX, phi, lower=True, left_side=True)

        Wb = v2.T @ v2 + 1./lambda_cap * jnp.identity(nbasis)

        chol_Wb = jax.lax.linalg.cholesky(Wb)  # Wb = Lb Lb^T

        # beta_hat = W_\beta^{-1} v1 => beta_hat = Lb^T\(Lb\y)
        beta_hat = jax.lax.linalg.triangular_solve(
            chol_Wb.T, 
            jax.lax.linalg.triangular_solve(chol_Wb, v1, lower=True,  left_side=True),
            lower=False,
            left_side=True)

        # y_residuals = y_train - \Phi(X_train)\hat{\beta}
        y_residuals = y_train[:,jnp.newaxis] - phi @ beta_hat

        #kinv_XX_res = \hat{K_y}^{-1} y_residuals
        kinv_XX_res = jax.lax.linalg.triangular_solve(
                    chol_XX.T, jax.lax.linalg.triangular_solve(chol_XX, y_residuals, lower=True,  
                                                               left_side=True),
                    left_side=True)

        #print('post train beta:',beta_hat)
        return beta_hat, kinv_XX_res

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


class GPEmuTraining():
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

    def __init__(self,
            kernel: Callable[[jnp.ndarray, 
                              jnp.ndarray, 
                              Dict[str, jnp.ndarray], 
                              float, float],jnp.ndarray] = kernel_RBF,
            var: float = 1e-5,
            order: int = 2,
            lambda_cap: float = 1.0,
            l_min: float = -5.0,
            l_max: float =  5.0,
            a_min: float =  0.0,
            a_max: float = 10.0,
            jitter: float = 1e-10,
            n_restart: int = 5,
            x_trans: bool = True,
            y_trans: bool = False,
            use_mean: bool = False) -> None:
                
        super().__init__(kernel,order,x_trans,y_trans,use_mean)

                        
        self.var = var        # "noise" level in principe can be set to very low level
        self.jitter = jitter  # to tweek choloesky algo add a small value on the diagonal if necessary
        self.order= order     # order of polynomial approx Phi(X)
        self.lambda_cap = lambda_cap # scale of polynomial coeff priors
        self.l_min = l_min # lower bound on kernel length (log scale)
        self.l_max = l_max # upper bound on kernel length (log scale)
        self.a_min = a_min # lower bound on kernel scale/amplitude (log scale)
        self.a_max = a_max # upper bound on kernel scale/amplitude (log scale)
        
        self.n_restart = n_restart #number of trials for kernel optimization in case of un-sucess


        #minimizer
        print_summary = False
        tol = 1e-30
        method = 'L-BFGS-B' # 'SLSQP'
        self.jscMin=jaxopt.ScipyBoundedMinimize(fun=self.marginal_likelihood,
                                          method=method, 
                                         tol=tol,
                                           options={'disp':print_summary,'ftol':tol, 'gtol':tol,
                                                    'maxiter':600})
        
        # init to None
        self.Init()
        
            
    def Init(self)->None:
        """
        Fnt used to reset internal data
        """
        self.theta   = None    # cosmo \Theta_i i<N_train
        self.y       = None    # f(\Theta_i)
        self.transform = None
        self.x_train = None    # Training input ( Ntrain x d ) with d: number of elements per train spl
        self.y_train = None    # Training target (Ntrain x 1)
        self.phi     = None    # Phi(X_train)
        self.nbasis  = None    # nber of elements Phi(X)  = 1+ order*d
        self.n_kern_params = None #  nber of Cosmo params + 1 = nber of K hyper-param
        
        # after kernel optimization (saved for prediction)
        self.kernel_hat  = None  # kernel parameters
        self.beta_hat    = None  # beta        
        self.kinv_XX_res = None  # K(Xtrain, Xtrain)^{-1} * (y_train - Phi(X_train) * beta_hat) 
        

    #setter necessary as jit function cannot set self.<variable> (leaks) (eg. post_training)
    #Todo: think to use  @property
    def set_kernel_hat(self, param_hat: Dict):
        self.kernel_hat = {'k_scale':param_hat[0],'k_length':param_hat[1:]}

    def set_beta_hat(self, x):
        self.beta_hat = x
    
    def set_kinv_XX_res(self, x):
        self.kinv_XX_res = x
    
    def mean_Theta_y(self,theta: jnp.ndarray, y: jnp.ndarray)->None:
        """
        Compute mean_theta and center or not output
        """
        self.theta  = theta       
        self.y      = y           # f(\Theta_i)

        d = self.theta.shape[1]   # the dimension of the problem
        self.n_trains = self.theta.shape[0] # the number of training point
        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'
        # the number of training points is greater than the number of dimension
        assert self.n_trains > d, msg

        # compute mean of training set
        self.mean_theta = jnp.mean(theta, axis=0)
        # centre the input on zero
        self.theta = theta - self.mean_theta
        
        # if we want to centre the output on zero
        if self.use_mean:
            self.mean_function = jnp.mean(y)
        else:
            self.mean_function = jnp.zeros(1)

        # the output is of size (ntrain x 1)
        self.y = y.reshape(self.n_trains, 1) - self.mean_function


        
    def do_transformation(self) -> None:
        '''
        Perform all transformations on input theta, y to get x_train, y_train
        '''
        # we transform both x and y if specified
        if (self.x_trans and self.y_trans):
            self.transform = transformation(self.theta, self.y)
            self.x_train = self.transform.x_transform()
            self.y_train = self.transform.y_transform()

        # we transform x only if specified
        elif self.x_trans:
            self.transform = transformation(self.theta, self.y)
            self.x_train = self.transform.x_transform()
            self.y_train = self.y

        # we keep the inputs and outputs (original basis)
        else:
            self.x_train = self.theta
            self.y_train = self.y
            
        #
        self.y_train = self.y_train.squeeze()   # Todo: a voir si ensuite ajouter un indice n'est pas superflu


    
    def basis_prior(self, par: Dict[str,Number])->Dict[str,jnp.array]:
        """
        beta ~ N(mu,C) with mu=0, C=lambda_cap * Id
        """
        
        n = par["nbasis"]
        lambda_cap = par["lambda_cap"]

        assert lambda_cap>0, f"lambda_cap should be >0 got {lambda_cap}" 

        beta = numpyro.sample("beta",
                              dist.MultivariateNormal(
                                  loc=jnp.zeros((n,)), 
                                    covariance_matrix=lambda_cap * jnp.identity(n), 
                                    precision_matrix = 1./lambda_cap * jnp.identity(n),
                              ),
                            )
        return {"beta": beta[:,jnp.newaxis]}
    
    
    def kernel_prior(self)->Dict[str,jnp.array]:
        """
        Kernel hyper-parameter priors
        k_scale (aka amplitude): 1 parameter
        k_length: as many as comosmological parameter
        """
                
        k_scale = numpyro.sample("k_scale", 
                                         dist.TransformedDistribution(
                                             dist.Uniform(low=self.a_min, high=self.a_max),
                                             dist.transforms.ExpTransform())
                                        )
        bnd = jnp.repeat(jnp.array([[self.l_min, self.l_max]]), self.n_lengthes, axis=0)
        k_length= numpyro.sample("k_length",
                                         dist.TransformedDistribution(
                                             dist.Uniform(bnd[:,0], high=bnd[:,1]),
                                             dist.transforms.ExpTransform())
                                        )
        

        
        return {"k_scale":k_scale, "k_length":k_length}

    
    
    def kernel_prior_UniformLin(self)->Dict[str,jnp.array]:
        """
        Kernel hyper-parameter priors
        k_scale (aka amplitude): 1 parameter
        k_length: as many as comosmological parameter
        """
        
        k_scale = numpyro.sample("k_scale",dist.Uniform(jnp.exp(self.a_min),jnp.exp(self.a_max)))
        #tmp = jnp.repeat(jnp.array([[0., 1.0]]), self.n_lengthes, axis=0)
        #k_length = numpyro.sample("k_length", dist.LogNormal(tmp[:,0],tmp[:,1]))

        tmp = jnp.repeat(jnp.exp(jnp.array([[self.l_min, self.l_max]])), self.n_lengthes, axis=0)
        k_length = numpyro.sample("k_length", dist.Uniform(tmp[:,0],tmp[:,1]))
        
        return {"k_scale":k_scale, "k_length":k_length}

    

    def prepare_training(self)->None:
        """
        Before training/optimization
        """
        self.phi    = self.compute_basis(self.x_train)   # (N x m) with m= order * n_cosmo +1)
        self.nbasis = self.phi.shape[1]       # m
        self.par_basis_prior = {"nbasis":self.nbasis, "lambda_cap": self.lambda_cap}

        self.n_lengthes = self.x_train.shape[1]
        self.n_kern_params = self.n_lengthes + 1
        

    
    def marginal_likelihood(self, params:jnp.array) -> float:
        """
        Marginal Likelihood with hypothesis of Gaussian Prior for beta
        """
        return -1.0 * _marginal_likelihood(params, 
                                    self.kernel,
                                    self.lambda_cap, 
                                    self.phi, 
                                    self.x_train, 
                                    self.var, 
                                    self.y_train, 
                                    self.n_trains)   #jitted helper

        
    def optimize(self, rng_key:jnp.array, init_param=None) -> bool:
        
        
        #Todo: this bounds can be done once
        bnd_low = [self.a_min]+[self.l_min]*self.n_lengthes
        bnd_high =[self.a_max]+[self.l_max]*self.n_lengthes
        bnd_low = jnp.exp(jnp.array(bnd_low))
        bnd_high = jnp.exp(jnp.array(bnd_high))
        #        print(f"bnd_low: {bnd_low}")
        #        print(f"bnd_high: {bnd_high}")
        
        # initial guess
        if init_param is None:
            rng_key, new_key = jax.random.split(rng_key)
            init_par = trace(seed(self.kernel_prior,new_key)).get_trace()
            init_param = jnp.hstack((init_par['k_scale']['value'], init_par['k_length']['value']))

        # print(f'init_param: {init_param}')
        
        # do minimization
        res = self.jscMin.run(init_param,bounds=(bnd_low, bnd_high))   # ScipyBoundedMinimize

        # optimized parameters and status
        param_hat, state = res
        
        return param_hat, self.marginal_likelihood(param_hat) #state.value ()  #state.fun_val (ScipyBoundedMinimize)

    def post_training(self)->None:
        
        self.beta_hat, self.kinv_XX_res= _post_training(self.kernel,
                              self.nbasis,
                              self.x_train,
                              self.kernel_hat,
                              self.var,
                              self.jitter, 
                              self.y_train, 
                              self.phi, 
                              self.lambda_cap
                              )

    
    
    def store_info(self, folder_name: str, file_name: str)->None:
        """
        store information for post traininf prediction
        """
        # let's try uncompressed 
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        fname = folder_name + '/' + file_name + '.npz'
        print(f"save in <{fname}>")
        
        np.savez(fname,
                mean_theta=self.mean_theta,
                x_trans= np.array([self.x_trans]),
                y_trans= np.array([self.y_trans]),
                mu_matrix= self.transform.mu_matrix if self.x_trans else np.array([[1.0]]),
                kernel_hat= self.kernel_hat,
                beta_hat= self.beta_hat,
                kinv_XX_res= self.kinv_XX_res,
                mean_function= self.mean_function,
                x_train=self.x_train,
                y_train=self.y_train,   #not really necessary
                y_min=np.array([self.transform.y_min])
                )

                

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
