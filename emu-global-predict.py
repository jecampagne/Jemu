# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: jaxcosmo
#     language: python
#     name: jaxcosmo
# ---

# +
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 20
import matplotlib.patches as mpatches

from timeit import default_timer as timer

import my_settings  as st         # configuration file
from helper import *              # load/save   
import util as ut              # utility function
# Simple non zero mean Gaussian Process class adapted for the emulator
from gaussproc_emu import *
# -

# # Predict all...

# # Start by loading the training results

root_dir = "./"

cosmologies = load_arrays(root_dir + 'trainingset/components', 'cosmologies')
print(f"Cosmo: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")

# +
# Growth factor
growth_factor = load_arrays(root_dir + 'trainingset/components', 'growth_factor')
print(f"Growth Fact: nber of training points {growth_factor.shape[0]} for {growth_factor.shape[1]} z (linear)")
n_gf = growth_factor.shape[1]
assert n_gf == st.nz, "Growth: Hummm something strange..."
print(f"The number of GPs to model Growth={n_gf} (= nber of z_bins) ")
folder_gf = root_dir + '/pknl_components' + st.d_one_plus + '/gf'
arg_gf = [[cosmologies, growth_factor[:, i], st.gf_args, folder_gf, 'gp_' + str(i)] for i in range(n_gf)]


# Pk Linear 
pk_linear = load_arrays(root_dir +'trainingset/components', 'pk_linear')
print(f"Linear Pk: nber of training points {pk_linear.shape[0]} for {pk_linear.shape[1]} k (log)")
n_pl = pk_linear.shape[1]
assert n_pl == st.nk, "Pk lin: Hummm something strange..."
print(f"The number of GPs to model Pklin={n_pl} (= nber of k_bins) ")
folder_pl = root_dir + '/pknl_components' + st.d_one_plus + '/pl'
arg_pl = [[cosmologies, pk_linear[:, i], st.pl_args, folder_pl, 'gp_' + str(i)] for i in range(n_pl)]

# Q-func (= 1+q)
q_function = load_arrays(root_dir +'trainingset/components', 'q_function')
print(f"Q-func: nber of training points {q_function.shape[0]} for {q_function.shape[1]} z x k(lin,log)")
n_qf = q_function.shape[1]
assert n_qf == st.nk * st.nz, "Qfunc: Hummm something strange..."
print(f"The number of GPs to model Q-func={n_qf}")
folder_qf = root_dir + '/pknl_components' + st.d_one_plus + '/qf'
if st.emu_one_plus_q:
    arg_qf = [[cosmologies, 1.0 + q_function[:, i], st.qf_args, folder_qf, 'gp_' + str(i)] for i in range(n_qf)]
else:
    arg_qf = [[cosmologies, q_function[:, i], st.qf_args, folder_qf, 'gp_' + str(i)] for i in range(n_qf)]
# -

# # Test with 1 new set of cosmo para.

# + tags=[]
par = {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}

theta_star = jnp.array([val for val in par.values()])

#Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h
theta_star
# -

# ## Growth factor

#y_trans = False
st.gf_args['y_trans']

gf_model = GPEmu(kernel=kernel_RBF,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.gf_args['y_trans'],
                         use_mean=st.use_mean)


pred_gf = jnp.zeros(n_gf)
for i_gf in range(n_gf):
    gf_model.load_info( arg_gf[i_gf][3], arg_gf[i_gf][4])
    pred_gf_cur = gf_model.simple_predict(theta_star)
    #print(f"[{i_gf}]",gf_model.kernel_hat, gf_model.beta_hat, pred_gf_cur)
    pred_gf= pred_gf.at[i_gf].set(pred_gf_cur)

pred_gf

# +
# training redshifts
zs=jnp.linspace(st.zmin, st.zmax, st.nz, endpoint=True)

z_new = jnp.linspace(st.zmin, st.zmax, 200, endpoint=True)
gf_new = jax.numpy.interp(z_new, zs, pred_gf)

plt.scatter(zs, pred_gf, label=r"$D(\Theta_\ast, z_{train})$")
plt.plot(z_new,gf_new,c="r", label=r"$D(\Theta_\ast,z_\ast)$")
plt.legend()
plt.xlabel("z")
plt.title("Growth Factor")
plt.grid();
# -

# # Linear Pk

st.pl_args['y_trans']

pl_model = GPEmu(kernel=kernel_RBF,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.pl_args['y_trans'],
                         use_mean=st.use_mean)


pred_pl = jnp.zeros(n_pl)
for i_pl in range(n_pl):
    pl_model.load_info(arg_pl[i_pl][3], arg_pl[i_pl][4])
    pred_pl_cur = pl_model.pred_original_function(theta_star)
    #print(f"[{i_gf}]",gf_model.kernel_hat, gf_model.beta_hat, pred_gf_cur)
    pred_pl= pred_pl.at[i_pl].set(pred_pl_cur)

# training k
ks = np.geomspace(st.k_min_h_by_Mpc, st.kmax, st.nk, endpoint=True)

k_new = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, 200, endpoint=True)
pl_new = jax.numpy.interp(k_new, ks, pred_pl)

plt.figure(figsize=(8,8))
plt.scatter(ks, pred_pl, s=3,label=r"$Pk_{lin}(\Theta_\ast, k_{train})$")
plt.plot(k_new,pl_new,c="r", label=r"$Pk_{lin}(\Theta_\ast,k_\ast)$")
plt.legend()
plt.xlabel("k")
plt.xscale("log")
plt.yscale("log")
plt.title(r"$Pk_{lin} (z=0)$")
plt.ylim([5e-4,2e5])
plt.grid();

# # Q-func

st.qf_args['y_trans']

qf_model = GPEmu(kernel=kernel_RBF,
                         order=st.order,
                         x_trans=st.x_trans,
                         y_trans=st.qf_args['y_trans'],          ######
                         use_mean=st.use_mean)


pred_qf = jnp.zeros(n_qf)
for i_qf in range(n_qf):
    qf_model.load_info(arg_qf[i_qf][3], arg_qf[i_qf][4])
    pred_qf_cur = qf_model.pred_original_function(theta_star)
    pred_qf= pred_qf.at[i_qf].set(pred_qf_cur)

pred_qf = pred_qf.reshape(st.nk, st.nz)

pred_qf.shape

# +
#interp2d (x,y,xp,yp,zp)
#    Args:
#        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
#            coordinates will be clamped to lie in-bounds.
#        xp, yp: 1D arrays of points specifying grid points where function values
#            are provided.
#       zp: 2D array of function values. For a function `f(x, y)` this must
#            satisfy `zp[i, j] = f(xp[i], yp[j])`
#    Returns:
#        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
# -

# qf_new[i] = qf_interp[k_new[i],z_new[i]]; len(k_new)=len(z_new) 
qf_new=ut.interp2d(k_new,z_new,ks,zs,pred_qf)   

# # Pk non linear at z=0.5  pour $\Theta_\ast$

z_star = 0.5
D_star = jax.numpy.interp(z_star, zs, pred_gf)

D_star

k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, 10000, endpoint=True)
pl_star_z0 = jax.numpy.interp(k_star, ks, pred_pl)
z_star = jnp.array([z_star]*k_star.shape[0])
qf_star = ut.interp2d(k_star,z_star,ks,zs,pred_qf)   

pl_star = D_star * pl_star_z0
pnl_star = pl_star * qf_star 

plt.figure(figsize=(8,8))
plt.scatter(k_star,pl_star, s=3, label=r"$P_{lin}(k, \Theta_\ast)$")
plt.scatter(k_star,pnl_star, s=3,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(r"Jemu @ $z=0.5$");


class emuPk():
    def __init__(self,):
        print("New Pk Emulator")
        
        self.z_train = jnp.linspace(st.zmin, st.zmax, st.nz, endpoint=True)
        self.k_train = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, st.nk, endpoint=True)

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
            gf_model = GPEmu(kernel=kernel_RBF,
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
            pl_model = GPEmu(kernel=kernel_RBF,
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
            qf_model = GPEmu(kernel=kernel_RBF,
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
        
    def interp_pk(self,theta_star, k_star, z_star):
        
        """
        interpolation non linear power spectrum aka pk_nl (growth:gf & linear power spec.: pk_l)
 
        input:
         theta_star (1D array)
            eg. array([0.12,0.022, 2.9, 1.0, 0.75])
            for {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}
         k_star (1D array)
            eg. k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, N, endpoint=True)
         z_star float
            
        return 
           array: pk_nl(theta_star, k_star[i], z_star[i])  i<N
           idem   gf and pk_l
        """
        pred_pnl, pred_gf, pred_pl_z0 = self.gp_kzgrid_pred(theta_star)

        interp_gf    = jax.numpy.interp(z_star, self.z_train, pred_gf)
        interp_pl_z0 = jax.numpy.interp(k_star, self.k_train, pred_pl_z0)
        z_star = jnp.array([z_star]*k_star.shape[0])
        interp_pnl   = ut.interp2d(k_star,z_star,self.k_train,self.z_train,pred_pnl)
        
        return interp_pnl, interp_gf, interp_pl_z0

emu = emuPk()

emu.load_all_gps(directory = root_dir + '/pknl_components' + st.d_one_plus)

par = {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}
theta_star = jnp.array([val for val in par.values()])
#Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h
theta_star

pred_gf, pred_pl, pred_qf = emu.gp_kzgrid_pred(theta_star)

pred_gf.shape,pred_pl.shape,pred_qf.shape

N=10*st.nk 
k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, N, endpoint=True)
z_star = 0.5
pk_nl, gf, pk_lz0 = emu.interp_pk(theta_star, k_star,z_star)

plt.figure(figsize=(8,8))
plt.scatter(k_star,pk_lz0*gf, s=3, label=r"$P_{lin}(k, \Theta_\ast)$")
plt.scatter(k_star,pk_nl, s=3,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"Jemu @ $z={z_star}$");

N=10*st.nk 
k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, N, endpoint=True)
z_star = 1.5
pk_nl, gf, pk_lz0 = emu.interp_pk(theta_star, k_star,z_star)

plt.figure(figsize=(8,8))
plt.scatter(k_star,pk_lz0*gf, s=3, label=r"$P_{lin}(k, \Theta_\ast)$")
plt.scatter(k_star,pk_nl, s=3,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"Jemu @ $z={z_star}$");


