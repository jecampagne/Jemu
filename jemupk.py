

import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial       ##### New approach: Custom PyTree


from jax_cosmo.core import Cosmology as jc_cosmo

#########
# JEC version > 3 July 2022 
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





#@jit JEC 4/7/22 jit this function leads to very long compilation time
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

    # Growth @ k_i, z_j
    ##pred_gf = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_gf ))
    pred_gf = jnp.array([gp.predict(theta_star) for gp in  gps_gf])
    pred_gf = pred_gf.reshape(jemu_st.nk,jemu_st.nz)

    # Linear Pk @ k_i, z=0
    ##pred_pl_z0 = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_pl))
    pred_pl_z0 = jnp.array([gp.predict(theta_star) for gp in  gps_pl])

    # Linear Pk @ k_i, z_j

    pred_pl = pred_pl_z0[:,jnp.newaxis] * pred_gf     # the shape is (Nk, Nz)

    return pred_pl


#@jit JEC 4/7/22 jit this function leads to very long compilation time
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


    # Non Linear Pk @ k_i, z=0
    ##pred_pnl_z0 = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_pnl ))
    pred_pnl_z0 = jnp.array([gp.predict(theta_star) for gp in gps_pnl])

    # Qfunc bis & k_i, z_j
    ## pred_qf =  jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star),gps_qf))
    pred_qf = jnp.array([gp.predict(theta_star) for gp in gps_qf])

    pred_qf = pred_qf.reshape(jemu_st.nk,jemu_st.nz)

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
