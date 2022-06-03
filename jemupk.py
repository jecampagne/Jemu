from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

import my_settings  as st         # configuration file
import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *

class JemuPk():
    def __init__(self,kernel_gf=kernel_RBF, kernel_pklin=kernel_RBF, kernel_qfunc=kernel_Matern12):
        print("New Pk Emulator")
        
        self.z_train = jnp.linspace(st.zmin, st.zmax, st.nz, endpoint=True)
        self.k_train = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, st.nk, endpoint=True)
        self.kernel_gf = kernel_gf
        self.kernel_pklin = kernel_pklin
        self.kernel_qfunc = kernel_qfunc

    def load_all_gps(self, directory: str):
        """
        Load optimizer GPs
        gf: growth factor D(z)
        pl: linear power spectrum P(k,z=0)
        qf: q-function (1+q(k,z))
        """
        
        # Growth factor
        folder_gf = directory + '/gf'
        n_gf = st.nz
        arg_gf = [[folder_gf, 'gp_' + str(i)] for i in range(n_gf)]
        
        self.gps_gf = []
        for i_gf in range(n_gf):
            gf_model = GPEmu(kernel=self.kernel_gf,     #kernel_RBF,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.gf_args['y_trans'],
                         use_mean=st.use_mean)
            gf_model.load_info(arg_gf[i_gf][0], arg_gf[i_gf][1])
            self.gps_gf.append(gf_model)
            
        # Linear Pk
        folder_pl = directory + '/pl'
        n_pl = st.nk
        arg_pl = [[folder_pl, 'gp_' + str(i)] for i in range(n_pl)]
        
        self.gps_pl = []
        for i_pl in range(n_pl):
            pl_model = GPEmu(kernel=self.kernel_pklin,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.pl_args['y_trans'],
                         use_mean=st.use_mean)
            pl_model.load_info(arg_pl[i_pl][0], arg_pl[i_pl][1])
            self.gps_pl.append(pl_model)
        
        #Q-func
        folder_qf = directory + '/qf'
        n_qf = st.nz * st.nk
        arg_qf = [[folder_qf, 'gp_' + str(i)] for i in range(n_qf)]
        
        self.gps_qf= []
        for i_qf in range(n_qf):
            qf_model = GPEmu(kernel=self.kernel_qfunc,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.qf_args['y_trans'],          ######
                         use_mean=st.use_mean)
            qf_model.load_info(arg_qf[i_qf][0], arg_qf[i_qf][1])
            self.gps_qf.append(qf_model)
            
            
    def gp_kzgrid_pred(self,theta_star):
        """
        Predict GPs at the (k_i,z_j) 'i,j' in st.nk x st.nz grid
        k_i = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, st.nk, endpoint=True)
        z_j = jnp.linspace(st.zmin, st.zmax, st.nz, endpoint=True)
        
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
        pred_pl = jnp.dot(pred_pl_z0.reshape(st.nk,1), pred_gf.reshape(1,st.nz))
        
        # Q-func @ (k_i, z_j)
        pred_qf = jnp.stack([qf_model.pred_original_function(theta_star) for qf_model in self.gps_qf])

        #Non linear Pk @ (k_i,z_j)
        pred_pnl = pred_qf.reshape(st.nk, st.nz) * pred_pl
        
        
        return pred_pnl, pred_gf, pred_pl_z0
    
    def interp_pk(self,theta_star, k_star, z_star, grid_opt=False):
        
        """
        interpolation non linear power spectrum aka pk_nl (growth:gf & linear power spec.: pk_l)
 
        input:
         theta_star (1D array)
            eg. array([0.12,0.022, 2.9, 1.0, 0.75])
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}
            or array([0.12,0.022, 0.8, 1.0, 0.75])  
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'sigma8': 0.8, 'n_s': 1.0, 'h': 0.75}
         k_star : see below 
         z_star :  idem 
         
         grid_opt: True if one wants to compute interpolations on a grid of (k_star,z_star)
                   False by default
                   
         return: interpolation at k*,z* of pknl(k*,z*), growth(z*), pklin(k*) @ z=0
         
         k*     z*     pknl shape       gf shape       pklin shape      comment
         ----------------------------------------------------------------------
         float float      (1,)            (1,)             (1,)         
         float  (n,)      (n,1)           (n,)             (1,)
         (n,)   float     (1,n)           (1,)             (n,)
         (n,)   (n,)      (n,)            (n,)             (n,)          grid_opt=False, Pknl(ki,zi)
         (m,)   (n,)      (n,m)           (n,)             (m,)          grid_opt=True, Pknl(kj,zi)
                                            
        """
        pred_pnl, pred_gf, pred_pl_z0 = self.gp_kzgrid_pred(theta_star)

        
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
            interp_pnl   = ut.interp2d(k_star_flat,z_star_flat,self.k_train,self.z_train,pred_pnl)
            interp_pnl = interp_pnl.reshape(z_star.shape[0],k_star.shape[0])
        else:
            interp_pnl   = ut.interp2d(k_star,z_star,self.k_train,self.z_train,pred_pnl)
        
        return interp_pnl, interp_gf, interp_pl_z0