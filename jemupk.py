from jax_cosmo.core import Cosmology as jc_cosmo
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *

class JemuPk():
    def __init__(self,param_train):
        #print("New Pk Emulator")
        self.kernel_gf    = param_train['kernel_gf']   # GP kernel for the Growth factor
        self.kernel_pklin = param_train['kernel_pklin'] # GP kernel for the Pk_lin(k,z=0)
        self.kernel_qfunc = param_train['kernel_qfunc'] # GP kernel for the q(k,z) (non linear part)
        
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

        self.x_trans = param_train['x_trans']           # GP input transformation
        self.gf_y_trans = param_train['gf_y_trans']     # GP ouput tranformation Growth factor
        self.pl_y_trans = param_train['pl_y_trans']     # idem linear Pk
        self.qf_y_trans = param_train['qf_y_trans']     # idem qfunc

        self.use_mean = param_train['use_mean']         # optional centering of the output
        
        ### Load optimised Emulator parameters ready for prediction
        
        self.gps_gf,self.gps_pl,self.gps_qf = self.load_all_gps(param_train['load_dir'])
        
        # Create a workspace where functions can store some precomputed results
        self._workspace = {}


    def load_all_gps(self, directory: str):
        """
        Load optimizer GPs
        gf: growth factor D(z)
        pl: linear power spectrum P(k,z=0)
        qf: q-function (1+q(k,z))
        """
        
        # Growth factor
        folder_gf = directory + '/gf'
        n_gf = self.nz
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
            
        # Linear Pk
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
        
        #Q-func
        folder_qf = directory + '/qf'
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
            
        return gps_gf, gps_pl, gps_qf
            

    def _gp_kzgrid_pred_testws(self,theta_star):
        """
        Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
        k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
        z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)
        
        input: theta_star (1D array)
            eg. array([0.12,0.022, 2.9, 1.0, 0.75])
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}
            
        return: growth factor, non linear pk, linear pk (without growth included)
        """

        #Check if the set of cosmo param (theta_star) has JUST been already treated 
        #Nb: we do not store in _workspace all the previous theta_star depend GPs data
        
        
        if not ('theta_star' in self._workspace and jnp.all(self._workspace['theta_star']==theta_star)):
            
            # Growth @ z_j
            pred_gf = jnp.stack([gf_model.simple_predict(theta_star) for gf_model in self.gps_gf])
            # Linear Pk @ k_i, z=0
            pred_pl_z0 = jnp.stack([pl_model.pred_original_function(theta_star) for pl_model in self.gps_pl])

            #Linear Pk @ (k_i,z_j)
            pred_pl = jnp.dot(pred_pl_z0.reshape(self.nk,1), pred_gf.reshape(1,self.nz))

            # Q-func @ (k_i, z_j)
            pred_qf = jnp.stack([qf_model.pred_original_function(theta_star) for qf_model in self.gps_qf])

            #Non linear Pk @ (k_i,z_j)
            pred_pnl = pred_qf.reshape(self.nk, self.nz) * pred_pl
        
            self._workspace = {'theta_star':theta_star,
                               'pred_pnl':pred_pnl,
                               'pred_gf':pred_gf,
                               'pred_pl_z0':pred_pl_z0,
                               }
            
        else:
            pred_pnl   = self._workspace['pred_pnl']
            pred_gf    = self._workspace['pred_gf']
            pred_pl_z0 = self._workspace['pred_pl_z0']
        
        return pred_pnl, pred_gf, pred_pl_z0

    def _gp_kzgrid_pred_all(self,theta_star):
        """
            Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
            k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
            z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

            input: theta_star (1D array)
                eg. array([0.12,0.022, 2.9, 1.0, 0.75])
                for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

            return: growth factor, non linear pk, linear pk (without growth included)
        """

            
        # Growth @ z_j
        pred_gf = jnp.stack([gf_model.simple_predict(theta_star) for gf_model in self.gps_gf])
        # Linear Pk @ k_i, z=0
        pred_pl_z0 = jnp.stack([pl_model.pred_original_function(theta_star) for pl_model in self.gps_pl])
        
        #Linear Pk @ (k_i,z_j)
        pred_pl = jnp.dot(pred_pl_z0.reshape(self.nk,1), pred_gf.reshape(1,self.nz))

        # Q-func @ (k_i, z_j)
        pred_qf = jnp.stack([qf_model.pred_original_function(theta_star) for qf_model in self.gps_qf])

        #Non linear Pk @ (k_i,z_j)
        pred_pnl = pred_qf.reshape(self.nk, self.nz) * pred_pl
        
        
        return pred_pnl, pred_gf, pred_pl_z0
    
    
    def _gp_kzgrid_pred_linear(self,theta_star):
        """
            Predict GPs at the (k_i,z_j) 'i,j' in nk x nz grid
            k_i = jnp.geomspace(k_min_h_by_Mpc,k_max_h_by_Mpc, nk, endpoint=True)
            z_j = jnp.linspace(zmin, zmax, nz, endpoint=True)

            input: theta_star (1D array)
                eg. array([0.12,0.022, 2.9, 1.0, 0.75])
                for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

            return: growth factor, linear pk (without growth included)
        """

            
        # Growth @ z_j
        pred_gf = jnp.stack([gf_model.simple_predict(theta_star) for gf_model in self.gps_gf])
        # Linear Pk @ k_i, z=0
        pred_pl_z0 = jnp.stack([pl_model.pred_original_function(theta_star) for pl_model in self.gps_pl])

        return pred_gf, pred_pl_z0
        
        
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
        pred_gf, pred_pl_z0 = self._gp_kzgrid_pred_linear(theta_star)
        
        #Linear Pk @ (k_i,z_j)
        pred_pl = jnp.dot(pred_pl_z0.reshape(self.nk,1), pred_gf.reshape(1,self.nz))

        # Q-func @ (k_i, z_j)
        pred_qf = jnp.stack([qf_model.pred_original_function(theta_star) for qf_model in self.gps_qf])

        #Non linear Pk @ (k_i,z_j)
        pred_pnl = pred_qf.reshape(self.nk, self.nz) * pred_pl
        
        return pred_pnl
    
    
    def interp_pk(self,cosmo, k_star, z_star, grid_opt=False):
        
        """
        interpolation non linear power spectrum aka pk_nl (growth:gf & linear power spec.: pk_l)
 
        input:
         cosmo:
            if (1D array)
                or array([0.12,0.022, 0.8, 1.0, 0.75])  
                for {'omega_cdm': 0.12, 'omega_b': 0.022, 'sigma8': 0.8, 'n_s': 1.0, 'h': 0.75}
            
            else Jax-cosmo.Cosmology instance
            
            
         k_star : k in h/MPc see below 
         z_star : redshift   see below
         
         grid_opt: True if one wants to compute interpolations on a grid of (k_star,z_star)
                   False by default
                   
         return: interpolation at k*,z* of 
                 pknl(z*, k*),   WARNING order return Pknl(zi,kj)
                 growth(z*), 
                 pklin(k*) @ z=0
         
         
         k*     z*     pknl shape       gf shape       pklin shape      comment
         ----------------------------------------------------------------------
         float float      (1,)            (1,)             (1,)         
         float  (n,)      (n,1)           (n,)             (1,)
         (n,)   float     (1,n)           (1,)             (n,)
         (n,)   (n,)      (n,)            (n,)             (n,)          grid_opt=False, Pknl(zi, ki)
         (m,)   (n,)      (n,m)           (n,)             (m,)          grid_opt=True, Pknl(zi,kj)
                                            
        """
        
        if isinstance(cosmo, jc_cosmo):
            theta_star= self.builtTheta(jc_cosmo)
        else:
            theta_star = cosmo
            
        
        #Get for this theta_star (cosmo param.) the GPs evaluations at all (k_i,z_j) of training grid. 
        pred_pnl, pred_gf, pred_pl_z0 = self._gp_kzgrid_pred_all(theta_star)

        ####
        # Compute the growth factor, pk_lin, pk_nl at the given (k_star, z_star)
        # Nb. we do not store the results on self._workspace as for a given theta_star
        #     this function is certainly called once for all.
        ####
        
        z_star = jnp.atleast_1d(z_star)
        k_star = jnp.atleast_1d(k_star)
        
        spline1D_z = InterpolatedUnivariateSpline(self.z_train, pred_gf)
        interp_gf    = spline1D_z(z_star)

        spline1D_k = InterpolatedUnivariateSpline(self.k_train, pred_pl_z0)
        interp_pl_z0 = spline1D_k(k_star)

        if grid_opt:
            k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
            k_star_flat = k_star_g.reshape((-1,))
            z_star_flat = z_star_g.reshape((-1,))
            interp_pnl = ut.interp2d(k_star_flat,z_star_flat,
                                     self.k_train,self.z_train,pred_pnl)
            interp_pnl = interp_pnl.reshape(z_star.shape[0],k_star.shape[0])
        else:
            assert k_star.shape == z_star.shape, "k_star and z_star should have same size"
            interp_pnl   = ut.interp2d(k_star,z_star,self.k_train,self.z_train,pred_pnl)
        
        return interp_pnl, interp_gf, interp_pl_z0
    
    def builtTheta(self, cosmo):
        return jnp.array([cosmo.Omega_c * (cosmo.h**2),
                                  cosmo.Omega_b * (cosmo.h**2),
                                  cosmo.sigma8,
                                  cosmo.n_s,
                                  cosmo.h])
        
    
    def linear_pk(self,cosmo, k_star, z_star=0.0):
        """
        cosmo: jax-cosmo
        """
        #transform jax-cosmo into emulator input array
        theta_star = self.builtTheta(cosmo)
        
        #Get for this theta_star (cosmo param.) the GPs evaluations at all (k_i,z_j) of training grid. 
        pred_gf, pred_pl_z0 = self._gp_kzgrid_pred_linear(theta_star)
        
        z_star = jnp.atleast_1d(z_star)
        k_star = jnp.atleast_1d(k_star)
        
        
        spline1D_z = InterpolatedUnivariateSpline(self.z_train, pred_gf)
        interp_gf  = spline1D_z(z_star)

        spline1D_k   = InterpolatedUnivariateSpline(self.k_train, pred_pl_z0)
        interp_pl_z0 = spline1D_k(k_star)

        pk_lin = interp_gf[:,jnp.newaxis] @ interp_pl_z0[jnp.newaxis,:]
        return pk_lin.squeeze()
    
    def nonlinear_pk(self, cosmo, k_star, z_star=0.0):
        
        
        #transform jax-cosmo into emulator input array
        theta_star = self.builtTheta(cosmo)

        z_star = jnp.atleast_1d(z_star)
        k_star = jnp.atleast_1d(k_star)

        pred_pnl = self._gp_kzgrid_pred_nlinear(theta_star)

        k_star_g, z_star_g = jnp.meshgrid(k_star, z_star)
        k_star_flat = k_star_g.reshape((-1,))
        z_star_flat = z_star_g.reshape((-1,))
        interp_pnl   = ut.interp2d(k_star_flat,z_star_flat,self.k_train,self.z_train,pred_pnl)
        interp_pnl = interp_pnl.reshape(z_star.shape[0],k_star.shape[0])
        
        return interp_pnl.squeeze()