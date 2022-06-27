from jax_cosmo.core import Cosmology as jc_cosmo
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *

from jax import jit
from functools import partial
#########
# JEC version > 9 June 2022
#
# Code for Pk Linear  [ Pk Non Linear ] (Cosmo, k, z)
# using:
# D(k_i,z_j):= Pk Linear(k_i,z_j)/ Pk Linear(k_i,z=0)
# Pk Linear(k_i,z=0),
# Q(k_i,z_j) := Pk Non Linear(k_i,z_j)/ Pk Non Linear(k_i,z=0)
# Pk Non Linear(k_i,z=0),

# z_j & k_i in train param
#########



class JemuPk():
    def __init__(self,param_train):
        #print("New Pk Emulator")
        self.kernel_gf    = param_train['kernel_gf_kscale']   # GP kernel for the Growth factor D(k,z)
        self.kernel_pklin = param_train['kernel_pklin'] # GP kernel for the Pk_lin(k,z=0)
        self.kernel_pknl  = param_train['kernel_pknl']  # GP kernel for the Pk_nl(k,z=0) 
        
        self.kernel_qfunc = param_train['kernel_qfunc_bis'] # GP kernel for the q(k,z) (non linear part)
        
        # Training parameters
        zmin = param_train['zmin']    # redshift min
        zmax = param_train['zmax']    # redshift max
        self.nz   = param_train['nz'] # number of redshifts
        self.z_train = jnp.linspace(zmin,zmax,self.nz, endpoint=True)

        
        kmin = param_train['kmin']    # k min (h/Mpc) 
        kmax = param_train['kmax']    # k max (h/Mpc) 
        self.nk   = param_train['nk'] # number of ks
        self.k_train = jnp.geomspace(kmin,kmax,self.nk, endpoint=True)
        
        
        self.order   = param_train['order']  # order of the polynomial basis (mean of the GP)

        self.x_trans     = param_train['x_trans']            # GP input transformation
        
        self.gf_y_trans  = param_train['gf_kscale_y_trans']  # GP ouput tranformation Growth factor
        self.pl_y_trans  = param_train['pl_y_trans']         # idem linear Pk        
        self.pnl_y_trans = param_train['pnl_y_trans']        # idem Non linear Pk
        self.qf_y_trans  = param_train['qf_bis_y_trans']     # idem qfunc bis

        self.use_mean = param_train['use_mean']         # optional centering of the output
        
        ### Load optimised Emulator parameters ready for prediction
        
        self.gps_gf,self.gps_pl, self.gps_pnl, self.gps_qf = self.load_all_gps(param_train['load_dir'])
        

    def load_all_gps(self, directory: str):
        """
        Load optimizer GPs
        gf: growth factor D(k,z)
        pl: linear power spectrum P(k,z=0)
        """
        
        # Growth factor with k-scale
        folder_gf = directory + '/gf_kscale'
        n_gf = self.nk * self.nz
        arg_gf = [[folder_gf, 'gp_' + str(i)] for i in range(n_gf)]
        
        gps_gf=[]
        for i_gf in range(n_gf):
            gf_model = GPEmu(kernel=self.kernel_gf,
                         order=self.order,
                         x_trans=self.x_trans,
                         y_trans=self.gf_y_trans,
                         use_mean=self.use_mean)
            gf_model.load_info(arg_gf[i_gf][0], arg_gf[i_gf][1])
            gps_gf.append(gf_model)
            
        # Linear Pk at z=0
        folder_pl = directory + '/pl'
        n_pl = self.nk
        arg_pl = [[folder_pl, 'gp_' + str(i)] for i in range(n_pl)]
        
        gps_pl=[]
        for i_pl in range(n_pl):
            pl_model = GPEmu(kernel=self.kernel_pklin,
                         order=self.order,
                         x_trans=self.x_trans,
                         y_trans=self.pl_y_trans,
                         use_mean=self.use_mean)
            pl_model.load_info(arg_pl[i_pl][0], arg_pl[i_pl][1])
            gps_pl.append(pl_model)
        

        # Non Linear Pk at z=0
        folder_pnl = directory + '/pnl'
        n_pnl = self.nk
        arg_pnl = [[folder_pnl, 'gp_' + str(i)] for i in range(n_pnl)]
        
        gps_pnl=[]
        for i_pnl in range(n_pnl):
            pnl_model = GPEmu(kernel=self.kernel_pknl,
                         order=self.order,
                         x_trans=self.x_trans,
                         y_trans=self.pnl_y_trans,
                         use_mean=self.use_mean)
            pnl_model.load_info(arg_pnl[i_pnl][0], arg_pnl[i_pnl][1])
            gps_pnl.append(pnl_model)



        #Q-func bis = Pk_NL(k,z)/Pk_NL(k,z=0)
        folder_qf = directory + '/qf_bis'
        n_qf = self.nz * self.nk
        arg_qf = [[folder_qf, 'gp_' + str(i)] for i in range(n_qf)]
        
        gps_qf=[]
        for i_qf in range(n_qf):
            qf_model = GPEmu(kernel=self.kernel_qfunc,
                         order=self.order,
                         x_trans=self.x_trans,
                         y_trans=self.qf_y_trans,          ######
                         use_mean=self.use_mean)
            qf_model.load_info(arg_qf[i_qf][0], arg_qf[i_qf][1])
            gps_qf.append(qf_model)
            

        return gps_gf, gps_pl, gps_pnl, gps_qf
    
    def _gp_kzgrid_pred_linear(self,theta_star):
        """
            Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
            k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
            z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

            input: theta_star (1D array)
                eg. array([0.12,0.022, 2.9, 1.0, 0.75])
                for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

            return: linear pk (without growth included)
        """

            
        # Growth @ k_i, z_j
        pred_gf = jnp.stack([gf_model.predict(theta_star) for gf_model in self.gps_gf])
        pred_gf = pred_gf.reshape(self.nk, self.nz)
        
        # Linear Pk @ k_i, z=0
        pred_pl_z0 = jnp.stack([pl_model.predict(theta_star) for pl_model in self.gps_pl])

        # Linear Pk @ k_i, z_j

        pred_pl = pred_pl_z0[:,jnp.newaxis] * pred_gf     # the shape is (Nk, Nz)

        return pred_pl
        
    
    def _gp_kzgrid_pred_nlinear(self,theta_star):
        """
            Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
            k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
            z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

            input: theta_star (1D array)
                eg. array([0.12,0.022, 2.9, 1.0, 0.75])
                for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

            return: growth factor, non linear pk, linear pk (without growth included)
        """

        # Non Linear Pk @ k_i, z=0
        pred_pnl_z0 = jnp.stack([pnl_model.predict(theta_star) for pnl_model in self.gps_pnl])


        # Qfunc bis & k_i, z_j
        pred_qf = jnp.stack([qf_model.predict(theta_star) for qf_model in self.gps_qf])
        pred_qf = pred_qf.reshape(self.nk,self.nz)

        # Non Linear Pk @ k_i, z_j
        pred_pnl =  pred_pnl_z0[:,jnp.newaxis] * pred_qf     # the shape is (Nk, Nz)

        return pred_pnl
    
    
    
    def builtTheta(self, cosmo):
        """ 
            From jax-cosmo.core.Cosmology to emulator parameters
        """
        #return jnp.array([cosmo.Omega_c * (cosmo.h**2),
        #                          cosmo.Omega_b * (cosmo.h**2),
        #                          cosmo.sigma8,
        #                          cosmo.n_s,
        #                          cosmo.h])
        return jnp.array([cosmo.Omega_c,
                                  cosmo.Omega_b,
                                  cosmo.sigma8,
                                  cosmo.n_s,
                                  cosmo.h])
        
    
    def linear_pk(self,cosmo, k_star, z_star=0.0):
        """
        cosmo: jax-cosmo.core.Cosmology
        interpolate Pk_lin on a grid (k_i, z_j)
        return PkLin(z,k)  : WARNING the shape is (Nz,Nk)
        """
        #transform jax-cosmo into emulator input array
        theta_star = self.builtTheta(cosmo)
        
        #Get for this theta_star (cosmo param.) the GPs evaluations at all (k_i,z_j) of training grid. 
        pred_pl = self._gp_kzgrid_pred_linear(theta_star)
        
        z_star = jnp.atleast_1d(z_star)
        k_star = jnp.atleast_1d(k_star)
        k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
        k_star_flat = k_star_g.reshape((-1,))
        z_star_flat = z_star_g.reshape((-1,))
        
        interp_pl = ut.interp2d(k_star_flat,z_star_flat,self.k_train,self.z_train,pred_pl)
        interp_pl = interp_pl.reshape(z_star.shape[0],k_star.shape[0])

        return interp_pl.squeeze()    # Care: shape (Nz, Nk) 

    def nonlinear_pk(self, cosmo, k_star, z_star=0.0):
        
        
        #transform jax-cosmo into emulator input array
        theta_star = self.builtTheta(cosmo)

        z_star = jnp.atleast_1d(z_star)
        k_star = jnp.atleast_1d(k_star)

        pred_pnl = self._gp_kzgrid_pred_nlinear(theta_star)

        k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
        k_star_flat = k_star_g.reshape((-1,))
        z_star_flat = z_star_g.reshape((-1,))
        interp_pnl = ut.interp2d(k_star_flat,z_star_flat,self.k_train,self.z_train,pred_pnl)
        interp_pnl = interp_pnl.reshape(z_star.shape[0],k_star.shape[0])
        
        return interp_pnl.squeeze()  # Care: shape (Nz, Nk) 
