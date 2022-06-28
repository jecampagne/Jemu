from jax_cosmo.core import Cosmology as jc_cosmo


import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *

from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
#########
# JEC version > 25 June 2022 
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
                folder_gf = directory + '/gf_kscale'
                fname_gf  = 'gp_' + str(i)
                gf_model = GPEmu(kernel=jemu_st.kernel_gf,
                             order=jemu_st.order,
                             x_trans=jemu_st.x_trans,
                             y_trans=jemu_st.gf_y_trans,
                             use_mean=jemu_st.use_mean)
                gf_model.load_info(folder_gf, fname_gf)
                return gf_model


            gps_gf = load_parallel_gp(load_one_gf, n_gf)
 

            # Linear Pk at z=0
            n_pl = jemu_st.nk
            def load_one_pl(i):
                folder_pl = directory + '/pl'
                fname_gf  = 'gp_' + str(i)
                pl_model = GPEmu(kernel=jemu_st.kernel_pklin,
                             order=jemu_st.order,
                             x_trans=jemu_st.x_trans,
                             y_trans=jemu_st.pl_y_trans,
                             use_mean=jemu_st.use_mean)
                pl_model.load_info(folder_pl, fname_gf)
                return pl_model
            
            gps_pl = load_parallel_gp(load_one_pl, n_pl)


            # Non Linear Pk at z=0
            n_pnl = jemu_st.nk
            def load_one_pnl(i):
                folder_pnl = directory + '/pnl'
                fname_gf  = 'gp_' + str(i)
                pnl_model = GPEmu(kernel=jemu_st.kernel_pknl,
                             order=jemu_st.order,
                             x_trans=jemu_st.x_trans,
                             y_trans=jemu_st.pnl_y_trans,
                             use_mean=jemu_st.use_mean)
                pnl_model.load_info(folder_pnl, fname_gf)
                return pnl_model

            gps_pnl = load_parallel_gp(load_one_pnl, n_pnl)


            #Q-func bis = Pk_NL(k,z)/Pk_NL(k,z=0)
            n_qf = jemu_st.nz * jemu_st.nk
            def load_one_qf(i):
                folder_qf = directory + '/qf_bis'
                fname_gf  = 'gp_' + str(i)
                qf_model = GPEmu(kernel=jemu_st.kernel_qfunc,
                             order=jemu_st.order,
                             x_trans=jemu_st.x_trans,
                             y_trans=jemu_st.qf_y_trans,          ######
                             use_mean=jemu_st.use_mean)
                qf_model.load_info(folder_qf, fname_gf)
                return qf_model

            gps_qf = load_parallel_gp(load_one_qf, n_qf)

            # Save
            GP_factory._ws = {"gf": gps_gf, "pl":gps_pl, "pnl":gps_pnl, "qf":gps_qf}

        # use worksape
        return GP_factory._ws



## class GP_factory():
##     done = False    # become True when load done
##     _ws = {}        # workspace
##     @classmethod
##     def make(cls, directory=None):
        
##         if not GP_factory.done:
##             GP_factory.done = True
            
##             # Growth factor with k-scale
## ##             folder_gf = directory + '/gf_kscale'
## ##             n_gf = jemu_st.nk * jemu_st.nz
## ##             arg_gf = [[folder_gf, 'gp_' + str(i)] for i in range(n_gf)]

## ##             gps_gf=[]
## ##             for i_gf in range(n_gf):
## ##                 gf_model = GPEmu(kernel=jemu_st.kernel_gf,
## ##                              order=jemu_st.order,
## ##                              x_trans=jemu_st.x_trans,
## ##                              y_trans=jemu_st.gf_y_trans,
## ##                              use_mean=jemu_st.use_mean)
## ##                 gf_model.load_info(arg_gf[i_gf][0], arg_gf[i_gf][1])
## ##                 gps_gf.append(gf_model)

##             n_gf = jemu_st.nk * jemu_st.nz

##             def load_one_gf(i):
##                 folder_gf = directory + '/gf_kscale'
##                 fname_gf  = 'gp_' + str(i)
##                 gf_model = GPEmu(kernel=jemu_st.kernel_gf,
##                              order=jemu_st.order,
##                              x_trans=jemu_st.x_trans,
##                              y_trans=jemu_st.gf_y_trans,
##                              use_mean=jemu_st.use_mean)
##                 gf_model.load_info(folder_gf, fname_gf)
##                 return gf_model

##             def load_parallel_gf():
##                 with concurrent.futures.ThreadPoolExecutor() as executor:
##                     X = executor.map(load_one_gf, range(n_gf))
##                 return list(X)

##             gps_gf = load_parallel_gf()
                                  

##             # Linear Pk at z=0
##             folder_pl = directory + '/pl'
##             n_pl = jemu_st.nk
##             arg_pl = [[folder_pl, 'gp_' + str(i)] for i in range(n_pl)]

##             gps_pl=[]
##             for i_pl in range(n_pl):
##                 pl_model = GPEmu(kernel=jemu_st.kernel_pklin,
##                              order=jemu_st.order,
##                              x_trans=jemu_st.x_trans,
##                              y_trans=jemu_st.pl_y_trans,
##                              use_mean=jemu_st.use_mean)
##                 pl_model.load_info(arg_pl[i_pl][0], arg_pl[i_pl][1])
##                 gps_pl.append(pl_model)


##             # Non Linear Pk at z=0
##             folder_pnl = directory + '/pnl'
##             n_pnl = jemu_st.nk
##             arg_pnl = [[folder_pnl, 'gp_' + str(i)] for i in range(n_pnl)]

##             gps_pnl=[]
##             for i_pnl in range(n_pnl):
##                 pnl_model = GPEmu(kernel=jemu_st.kernel_pknl,
##                              order=jemu_st.order,
##                              x_trans=jemu_st.x_trans,
##                              y_trans=jemu_st.pnl_y_trans,
##                              use_mean=jemu_st.use_mean)
##                 pnl_model.load_info(arg_pnl[i_pnl][0], arg_pnl[i_pnl][1])
##                 gps_pnl.append(pnl_model)



##             #Q-func bis = Pk_NL(k,z)/Pk_NL(k,z=0)
##             folder_qf = directory + '/qf_bis'
##             n_qf = jemu_st.nz * jemu_st.nk
##             arg_qf = [[folder_qf, 'gp_' + str(i)] for i in range(n_qf)]

##             gps_qf=[]
##             for i_qf in range(n_qf):
##                 qf_model = GPEmu(kernel=jemu_st.kernel_qfunc,
##                              order=jemu_st.order,
##                              x_trans=jemu_st.x_trans,
##                              y_trans=jemu_st.qf_y_trans,          ######
##                              use_mean=jemu_st.use_mean)
##                 qf_model.load_info(arg_qf[i_qf][0], arg_qf[i_qf][1])
##                 gps_qf.append(qf_model)

##                 # Save
##                 GP_factory._ws = {"gf": gps_gf, "pl":gps_pl, "pnl":gps_pnl, "qf":gps_qf}

##         # use worksape
##         return GP_factory._ws



#@jit
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
    pred_gf = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_gf ))
    pred_gf = pred_gf.reshape(jemu_st.nk,jemu_st.nz)

    # Linear Pk @ k_i, z=0
    pred_pl_z0 = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_pl))

    # Linear Pk @ k_i, z_j

    pred_pl = pred_pl_z0[:,jnp.newaxis] * pred_gf     # the shape is (Nk, Nz)

    return pred_pl


#@jit
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
    #pred_pnl_z0 = jnp.stack([pnl_model.predict(theta_star) for pnl_model in gps_pnl])
    pred_pnl_z0 = jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star), gps_pnl ))


    # Qfunc bis & k_i, z_j
    #pred_qf = jnp.stack([qf_model.predict(theta_star) for qf_model in gps_qf])
    pred_qf =  jnp.array(jax.tree_map(lambda gp: gp.predict(theta_star),gps_qf))
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

def linear_pk(cosmo, k_star, z_star=0.0):
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


def nonlinear_pk(cosmo, k_star, z_star=0.0):


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
