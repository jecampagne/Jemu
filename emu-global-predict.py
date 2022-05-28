# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: jaxccl
#     language: python
#     name: jaxccl
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
plt.scatter(ks, pred_pl, s=10,label=r"$Pk_{lin}(\Theta_\ast, k_{train})$")
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

ks.shape,zs.shape

ks

zs

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
qf_new=ut.coogan_interp2d(k_new,z_new,ks,zs,pred_qf)   
qf_new_bis=jnp.array([ut.coogan_interp2d(k,z,ks,zs,pred_qf)  for k,z in zip(k_new,z_new)])
qf_new_jec=jnp.array([ut.jec_interp2d(k,z,ks,zs,pred_qf) for k,z in zip(k_new,z_new)])

plt.plot(qf_new, label="Coogan (a)")
plt.plot(qf_new_bis,ls="--", label="Coogan (b)")
plt.plot(qf_new_jec,ls="-.", label="JEC")
plt.xlim([140,160])
plt.ylim([0,10])
plt.grid()
plt.legend();

# # Pk non linear at z=0.5  pour $\Theta_\ast$

z_star = 0.5
D_star = jax.numpy.interp(z_star, zs, pred_gf)

D_star

ks.shape, zs.shape

z_star = 0.5
k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, 100, endpoint=True)
pl_star_z0 = jax.numpy.interp(k_star, ks, pred_pl)
#z_star = jnp.array([z_star]*k_star.shape[0])
qf_star = jnp.array([ut.coogan_interp2d(k,z_star,ks,zs,pred_qf)   for k in k_star])

# +
#qf_star
# -

qf_star.shape, D_star, pl_star_z0.shape

pl_star = D_star * pl_star_z0
pnl_star = pl_star * qf_star 

plt.figure(figsize=(8,8))
plt.scatter(k_star,pl_star, s=10, label=r"$P_{lin}(k, \Theta_\ast)$")
plt.scatter(k_star,pnl_star, s=10,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(r"Jemu @ $z=0.5$");

from jemupk import *

emu = JemuPk()

emu.load_all_gps(directory = root_dir + '/pknl_components' + st.d_one_plus)

par = {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}
theta_star = jnp.array([val for val in par.values()])
#Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h
theta_star

pred_gf, pred_pl, pred_qf = emu.gp_kzgrid_pred(theta_star)

pred_gf.shape,pred_pl.shape,pred_qf.shape

N=10*st.nk 
k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.kmax, N, endpoint=True)
z_star = 0.
pk_nl, gf, pk_lz0 = emu.interp_pk(theta_star, k_star,z_star)

plt.figure(figsize=(8,8))
plt.plot(k_star,pk_lz0*gf, lw=3, label=r"$P_{lin}(k, \Theta_\ast)$")
plt.plot(k_star,pk_nl, lw=1,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"Jemu @ $z={z_star}$");
plt.xlim([1e-3,1])
plt.ylim([1e2,1e5])


# # Comparaison 

import jax_cosmo as jc
import pyccl as ccl

# +
# Sur un jeu de theta cosmo de training
# -

cosmologies

#Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h
#cosmologies[0]=array([0.12616742, 0.02163407, 2.72607173, 1.15724975, 0.73022459])
h_emu = 0.73022459
omega_c_emu = 0.12616742    # omega_c h^2
omega_b_emu = 0.02163407   #omega_b h^2
n_s_emu = 1.15724975
ln1010As_emu = 2.72607173
As_emu = 10**(-10)*np.exp(ln1010As_emu)


# +
#h_emu = 0.75
#omega_c_emu = 0.12    # omega_c h^2
#omega_b_emu = 0.022   #omega_b h^2
#n_s_emu = 1.0
#ln1010As_emu=2.9
#As_emu = 10**(-10)*np.exp(ln1010As_emu)
# -


# ## Attention: l'emulateur prend 'Omega_c h^2' et 'Omega_b h^2' contrairement à CCL et jax-cosmo 

# +


omega_c_ccl = omega_c_emu/h_emu**2
omega_b_ccl = omega_b_emu/h_emu**2


sigma8_emu = ccl.Cosmology(
    Omega_c=omega_c_ccl, Omega_b=omega_b_ccl, 
    h=h_emu, A_s=As_emu, n_s=n_s_emu, Neff=0,
    transfer_function='boltzmann_camb').sigma8()

cosmo_ccl= ccl.Cosmology(
    Omega_c=omega_c_ccl, Omega_b=omega_b_ccl, 
    h=h_emu, sigma8=sigma8_emu, n_s=n_s_emu, Neff=0,
    transfer_function='eisenstein_hu', matter_power_spectrum='halofit')
# -

params_emu = {'omega_cdm': omega_c_emu, 'omega_b': omega_b_emu, 
             'ln10^{10}A_s':ln1010As_emu , 
             'n_s': n_s_emu, 'h': h_emu}
theta_star = jnp.array([val for val in params_emu.values()])
#Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h
theta_star

Nk=10*st.nk 
k_star = jnp.geomspace(5e-4, 5e1,Nk, endpoint=True)
z_star = 1.22631579
pk_nl, gf, pk_lz0 = emu.interp_pk(theta_star, k_star,z_star)

pk_nl /= h_emu**2
pk_lz0 /= h_emu**2

# +
cosmo_jax = jc.Cosmology(Omega_c=omega_c_ccl, Omega_b=omega_b_ccl, 
    h=h_emu, sigma8=sigma8_emu, n_s=n_s_emu, Omega_k=0, w0=-1.0,wa=0.0)

pk_lin_ccl = ccl.linear_matter_power(cosmo_ccl, k_star*cosmo_jax.h, 1./(1+z_star)) #last is scale factor 1=>z=0

pk_lin_jc = jc.power.linear_matter_power(cosmo_jax,k_star, 1./(1+z_star))/cosmo_jax.h**3
# -

plt.figure(figsize=(10,8))
plt.plot(k_star,pk_lz0*gf,lw=5, label="Jemu")
plt.plot(k_star,pk_lin_jc,lw=3, label="jax_cosmo")
plt.plot(k_star,pk_lin_ccl,lw=3, ls="--",label="ccl")
#plt.plot(k_star,pk_nl, lw=1,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"$z={z_star}$");
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])

# +
cosmo_ccl_EH= ccl.Cosmology(
    Omega_c=omega_c_ccl, Omega_b=omega_b_ccl, 
    h=h_emu, sigma8=sigma8_emu, n_s=n_s_emu, Neff=0,
    transfer_function='eisenstein_hu', matter_power_spectrum='halofit')


cosmo_ccl_CAMB_nus= ccl.Cosmology(
    Omega_c=omega_c_ccl, Omega_b=omega_b_ccl, 
    h=h_emu, sigma8=sigma8_emu, n_s=n_s_emu, m_nu=0.06, Neff=3.046,
    transfer_function='boltzmann_camb', matter_power_spectrum='halofit')


pk_nonlin_ccl_EH = ccl.nonlin_matter_power(cosmo_ccl_EH, k_star*cosmo_jax.h, 
                                        1./(1+z_star)) #last is scale factor 1=>z=0

pk_nonlin_ccl_CAMB_nus = ccl.nonlin_matter_power(cosmo_ccl_CAMB_nus, k_star*cosmo_jax.h, 
                                        1./(1+z_star)) #last is scale factor 1=>z=0

pk_nonlin_jc = jc.power.nonlinear_matter_power(cosmo_jax,k_star, 
                                               1./(1+z_star))/cosmo_jax.h**3
# -

plt.figure(figsize=(10,8))
plt.plot(k_star,pk_nl,lw=5, label="Jemu")
plt.plot(k_star,pk_nonlin_jc,lw=3, label="jax_cosmo")
plt.plot(k_star,pk_nonlin_ccl_EH,lw=3, ls="--",label="ccl (EH+Halofit)")
plt.plot(k_star,pk_nonlin_ccl_CAMB_nus,lw=3, ls="-.",label="ccl (CAMB)")
#plt.plot(k_star,pk_nl, lw=1,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"$z={z_star}$");
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])

thetabin=0
cosmologies[thetabin]

zbin=5
zs[zbin]

gF = growth_factor[thetabin,zbin]

pklinz0 = pk_linear[thetabin]

pklin= gF * pklinz0

Qf = (1+q_function[thetabin]).reshape(st.nk, st.nz)

Qf[:,zbin]

pknl = pklin * Qf[:,zbin]

# +
#rescaling
# -

h = cosmologies[thetabin][-1]

h

pklin /= h**2
pknl /= h**2

plt.scatter(ks,pklin,label='train Plin')
plt.scatter(ks,pknl,label='train Pnl')
plt.xscale("log")
plt.yscale("log")

# +
plt.figure(figsize=(10,8))
plt.plot(k_star,pk_nl,lw=5, label="Jemu")
plt.plot(k_star,pk_nonlin_jc,lw=3, label="jax_cosmo")
plt.plot(k_star,pk_nonlin_ccl_EH,lw=3, ls="--",label="ccl (EH+Halofit)")
#plt.scatter(ks,pklin,label='train Plin')
plt.scatter(ks,pknl,label='train Pnl')

#plt.plot(k_star,pk_nonlin_ccl_CAMB,lw=3, ls="--",label="ccl (CAMB)")
#plt.plot(k_star,pk_nl, lw=1,label=r"$P_{nl}(k, \Theta_\ast)$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.title(rf"$z={zs[zbin]:.4f}$, {params_emu}");
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])
# -




