import numpy as np 
# for I/O
import os          


import util as ut              # utility function

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap
import jax.numpy as jnp
#from functools import partial
from jax.tree_util import register_pytree_node_class, Partial       ##### New approach: Custom PyTree



from jax_cosmo.core import Cosmology as jc_cosmo
from jax_cosmo.scipy.interpolate import interp as jc_interp1d  #JEC 18/10/22

#########
# JEC version >= 5 July 2022 
#
# Code for Pk Linear  [ Pk Non Linear ] (Cosmo, k, z)
# using:
# D(k_i,z_j):= Pk Linear(k_i,z_j)/ Pk Linear(k_i,z=0)
# Pk Linear(k_i,z=0),
# Q(k_i,z_j) := Pk Non Linear(k_i,z_j)/ Pk Non Linear(k_i,z=0)
# Pk Non Linear(k_i,z=0),

# z_j & k_i in train param
# ########

from typing import NamedTuple, Any
import settings_gfpkq_120x20  as st         # configuration file (update 2/June/22)


import concurrent.futures  # JEC 27/6/22
from typing import Union, Dict, Callable, Optional, Tuple


#######################################
### Gaussian Process part
##         y(x) = \Phi(x).\beta + GP(0,K(x,x^\prime; \alpha)
##         \alpha = (amplitude, \{\ell \}_i<d)  with d dimension of x
##         K: GP kernel as RBF

##         Determination of \beta & \alpha based on: 
##         Gaussian process class
##         Using C. E. Rasmussen & C. K. I. Williams, 
##         Gaussian Processes for Machine Learning, the MIT Press, 2006 

##         2.7 Incorporating Explicit Basis Functions
##         Eqs. 2.39-2.41 (mean only)
##         Eq. 2.43 Marginal Likelihood
##         Hypothesis: beta prior=N(0,\lambda_cap Id) => simplififications
#########


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


predict = _pred_original_function       #### MAIN function called by _gp_kzgrid_pred_linear (and non linear equiv.)


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



#####
## Gaussian Process PyTree
#####
    

@register_pytree_node_class
class GPEmu:
    def __init__(self,
                 order, # : int
                 kernel, #: Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], float, float],jnp.ndarray]
                 x_train,  #: ndarray
                 mean_theta, # ndarray
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
        self.beta_hat = beta_hat
        self.kinv_XX_res = kinv_XX_res
        self.mean_function = mean_function
        self.mu_matrix = mu_matrix
        self.y_min = y_min
        
        

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


#######################################

class JemuSettings(NamedTuple):
    kernel_pklin : Any = kernel_RBF
    kernel_pknl : Any = kernel_RBF
    kernel_gf : Any = kernel_Matern12
    kernel_qfunc : Any = kernel_Matern12
    nz : int = st.nz
    z_train : jnp.array = jnp.linspace(st.zmin,st.zmax,st.nz, endpoint=True)
    nk: int = st.nk
    k_train : jnp.array = jnp.geomspace(st.k_min_h_by_Mpc,st.k_max_h_by_Mpc,st.nk, endpoint=True)
    order : int = st.order
    x_trans : bool = st.x_trans
    gf_y_trans : bool = st.gf_scale_args['y_trans']
    pl_y_trans : bool = st.pl_args['y_trans']
    pnl_y_trans : bool = st.pnl_args['y_trans']
    qf_y_trans : bool = st.qf_bis_args['y_trans']
    use_mean  : bool = st.use_mean


jemu_st = JemuSettings()

# Gauss Process factory

class GP_factory():
    done = False    # become True when load done
    _ws = {}        # workspace


    # JEC 18/10/22  Should be deprecated soon use load_gps
    @classmethod
    def make(cls, directory=None):

        if not GP_factory.done:
            GP_factory.done = True

            # Parallel execution with the maximum threads  JEC 28/6/22
            def load_parallel_gp(loader,n):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    X = executor.map(loader, range(n))
                return list(X)

            # Growth factor with k-scale
            n_gf = jemu_st.nk * jemu_st.nz

            def load_one_gf(i):
                folder_name = directory + '/gf_kscale'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_gf,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)
                gf_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )
                
                return gf_model


            gps_gf = load_parallel_gp(load_one_gf, n_gf)
 

            # Linear Pk at z=0
            n_pl = jemu_st.nk
            def load_one_pl(i):
                folder_name = directory + '/pl'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_pklin,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)
                pl_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,   ###
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )
                return pl_model
            
            gps_pl = load_parallel_gp(load_one_pl, n_pl)


            # Non Linear Pk at z=0
            n_pnl = jemu_st.nk
            def load_one_pnl(i):
                folder_name = directory + '/pnl'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_pknl,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)

                pnl_model = GPEmu(order=jemu_st.order,
                                  kernel=kernel, ####
                                  x_train=data["x_train"],
                                  mean_theta=data['mean_theta'],
                                  beta_hat=data["beta_hat"],
                                  kinv_XX_res=data["kinv_XX_res"],
                                  mean_function=data["mean_function"],
                                  mu_matrix=data['mu_matrix'],
                                  y_min=data['y_min'][0]
                                  )

                return pnl_model

            gps_pnl = load_parallel_gp(load_one_pnl, n_pnl)


            #Q-func bis = Pk_NL(k,z)/Pk_NL(k,z=0)
            n_qf = jemu_st.nz * jemu_st.nk
            def load_one_qf(i):
                folder_name = directory + '/qf_bis'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu

                kernel = Partial(jemu_st.kernel_qfunc,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)


                qf_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )

                return qf_model

            gps_qf = load_parallel_gp(load_one_qf, n_qf)

            # Save
            GP_factory._ws = {"gf": gps_gf, "pl":gps_pl, "pnl":gps_pnl, "qf":gps_qf}

        # use worksape
        return GP_factory._ws

    #JEC 18/10/22 test to load on demand
    @classmethod
    def load_gps(cls, directory=None, gp_names=["Pklin0","Growth","Pknl0","Qfunc"]):

        if not GP_factory.done:
            GP_factory.done = True

            # Parallel execution with the maximum threads  JEC 28/6/22
            def load_parallel_gp(loader,n):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    X = executor.map(loader, range(n))
                return list(X)

            # Growth factor with k-scale
            n_gf = jemu_st.nk * jemu_st.nz

            def load_one_gf(i):
                folder_name = directory + '/gf_kscale'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_gf,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)
                gf_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )
                
                return gf_model
 

            # Linear Pk at z=0
            n_pl = jemu_st.nk
            def load_one_pl(i):
                folder_name = directory + '/pl'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_pklin,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)
                pl_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,   ###
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )
                return pl_model
            

            # Non Linear Pk at z=0
            n_pnl = jemu_st.nk
            def load_one_pnl(i):
                folder_name = directory + '/pnl'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu
                kernel = Partial(jemu_st.kernel_pknl,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)

                pnl_model = GPEmu(order=jemu_st.order,
                                  kernel=kernel, ####
                                  x_train=data["x_train"],
                                  mean_theta=data['mean_theta'],
                                  beta_hat=data["beta_hat"],
                                  kinv_XX_res=data["kinv_XX_res"],
                                  mean_function=data["mean_function"],
                                  mu_matrix=data['mu_matrix'],
                                  y_min=data['y_min'][0]
                                  )

                return pnl_model



            #Q-func bis = Pk_NL(k,z)/Pk_NL(k,z=0)
            n_qf = jemu_st.nz * jemu_st.nk
            def load_one_qf(i):
                folder_name = directory + '/qf_bis'
                file_name  = 'gp_' + str(i)
                fname = folder_name + '/' + file_name + '.npz'  
                data = np.load(fname, allow_pickle=True)   ##  JEC 1/7/22 load here to instantiate GPEmu

                kernel = Partial(jemu_st.kernel_qfunc,
                                 params=data["kernel_hat"].item(),
                                 noise=0.0, jitter=0.0)


                qf_model = GPEmu(order=jemu_st.order,
                                 kernel=kernel,
                                 x_train=data["x_train"],
                                 mean_theta=data['mean_theta'],
                                 beta_hat=data["beta_hat"],
                                 kinv_XX_res=data["kinv_XX_res"],
                                 mean_function=data["mean_function"],
                                 mu_matrix=data['mu_matrix'],
                                 y_min=data['y_min'][0]
                                 )

                return qf_model

            # ["Pklin0","Growth","Pknl0","Qfunc"]
            for name in gp_names:
                if name == "Pklin0":
                    GP_factory._ws["pl"] = load_parallel_gp(load_one_pl, n_pl)
                elif  name == "Growth":
                    GP_factory._ws["gf"] = load_parallel_gp(load_one_gf, n_gf)
                elif name == "Pknl0":
                    GP_factory._ws["pnl"] = load_parallel_gp(load_one_pnl, n_pnl)
                elif name == "Qfunc":
                    GP_factory._ws["qf"] = load_parallel_gp(load_one_qf, n_qf)
                else:
                    raise ValueError(f"gp_name <{name}> does not exists: choose among Pklin0,Growth,Pknl0,Qfunc")
                
        # use worksape
        assert len(GP_factory._ws) != 0, "Error empty workspace after load"
            
        return GP_factory._ws



#JEC 5/7/22
def pytrees_stack(pytrees, axis=0):
    results = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results

#JEC 18/10/22
@jit  
def _gp_kgrid_pred_linear_z0(theta_star):
    gps = GP_factory.load_gps()
    gps_pl = gps["pl"]

    func = lambda x : predict(x,theta_star)
    pred_pl_z0 = jax.vmap(func)(pytrees_stack(gps_pl))
    return pred_pl_z0


@jit  
def _gp_kzgrid_pred_linear(theta_star):
    """
        Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
        k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
        z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

        input: theta_star (1D array)
            eg. array([0.12,0.022, 2.9, 1.0, 0.75])
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

        return: linear pk (without growth included)
    """
    gps = GP_factory.make()
    gps_gf = gps["gf"]
    gps_pl = gps["pl"]

    func = lambda x : predict(x,theta_star)
    

    # Growth @ k_i, z_j
    pred_gf = jax.vmap(func)(pytrees_stack(gps_gf))
    pred_gf = pred_gf.reshape(jemu_st.nk,jemu_st.nz)

    # Linear Pk @ k_i, z=0
    ##pred_pl_z0 = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_pl))
    ## pred_pl_z0 = jnp.array([gp.predict(theta_star) for gp in  gps_pl])
    pred_pl_z0 = jax.vmap(func)(pytrees_stack(gps_pl))

    # Linear Pk @ k_i, z_j

    pred_pl = pred_pl_z0[:,jnp.newaxis] * pred_gf     # the shape is (Nk, Nz)

    return pred_pl


@jit
def _gp_kzgrid_pred_nlinear(theta_star):
    """
        Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
        k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
        z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

        input: theta_star (1D array)
            eg. array([0.12,0.022, 2.9, 1.0, 0.75])
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

        return: growth factor, non linear pk, linear pk (without growth included)
    """
    gps = GP_factory.make()
    gps_qf = gps["qf"]
    gps_pnl = gps["pnl"]


    func = lambda x : predict(x,theta_star)

    # Qfunc bis & k_i, z_j
    pred_qf = jax.vmap(func)(pytrees_stack(gps_qf))
    pred_qf = pred_qf.reshape(jemu_st.nk,jemu_st.nz)

    # Non Linear Pk @ k_i, z=0
    pred_pnl_z0= jax.vmap(func)(pytrees_stack(gps_pnl))


    # Non Linear Pk @ k_i, z_j
    pred_pnl =  pred_pnl_z0[:,jnp.newaxis] * pred_qf     # the shape is (Nk, Nz)

    return pred_pnl






def _builtTheta(cosmo):
    """ 
        From jax-cosmo.core.Cosmology to emulator parameters
    """
    return jnp.array([cosmo.Omega_c,
                              cosmo.Omega_b,
                              cosmo.sigma8,
                              cosmo.n_s,
                              cosmo.h])



def linear_pk(cosmo, k_star, z_star):
    """
    cosmo: jax-cosmo.core.Cosmology
    interpolate Pk_lin on a grid (k_i, z_j)
    return PkLin(z,k)  : WARNING the shape is (Nz,Nk)
    """
    #transform jax-cosmo into emulator input array
    theta_star = _builtTheta(cosmo)

    #Get for this theta_star (cosmo param.) the GPs evaluations at all (k_i,z_j) of training grid. 
    pred_pl = _gp_kzgrid_pred_linear(theta_star)

    z_star = jnp.atleast_1d(z_star)
    k_star = jnp.atleast_1d(k_star)
    k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
    k_star_flat = k_star_g.reshape((-1,))
    z_star_flat = z_star_g.reshape((-1,))

    interp_pl = ut.interp2d(k_star_flat,z_star_flat,jemu_st.k_train,jemu_st.z_train,pred_pl)
    interp_pl = interp_pl.reshape(z_star.shape[0],k_star.shape[0])

    return interp_pl.squeeze()    # Care: shape (Nz, Nk) 


#JEC 18/10/22: compute linear Pk at z=0 for a vector of k
def linear_pk_z0(cosmo, k_star):
    #transform jax-cosmo into emulator input array
    theta_star = _builtTheta(cosmo)
    #compute pk on the predefined k-grid
    pred_pl = _gp_kgrid_pred_linear_z0(theta_star)
    #1D interpolation
    interp_pl = jc_interp1d(k_star,jemu_st.k_train,pred_pl)
    
    return interp_pl
    



def nonlinear_pk(cosmo, k_star, z_star):


    #transform jax-cosmo into emulator input array
    theta_star = _builtTheta(cosmo)

    z_star = jnp.atleast_1d(z_star)
    k_star = jnp.atleast_1d(k_star)

    pred_pnl = _gp_kzgrid_pred_nlinear(theta_star)

    k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
    k_star_flat = k_star_g.reshape((-1,))
    z_star_flat = z_star_g.reshape((-1,))
    interp_pnl = ut.interp2d(k_star_flat,z_star_flat,jemu_st.k_train,jemu_st.z_train,pred_pnl)
    interp_pnl = interp_pnl.reshape(z_star.shape[0],k_star.shape[0])

    return interp_pnl.squeeze()  # Care: shape (Nz, Nk) 
