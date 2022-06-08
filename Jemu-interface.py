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

import numpy as onp
import jax.numpy as np
import jax_cosmo as jc      # Jax-cosmo lib
from jemupk import *        # Jax Emulator of CLASS
import settings_gfpkq  as st         # configuration file (update 2/June/22)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 20
import matplotlib.patches as mpatches

# +
param_emu = {}

param_emu['zmin'] = st.zmin
param_emu['zmax'] = st.zmax
param_emu['nz']   = st.nz

param_emu['kmin'] = st.k_min_h_by_Mpc
param_emu['kmax'] = st.k_max_h_by_Mpc
param_emu['nk']   = st.nk

param_emu['order'] = st.order
param_emu['x_trans'] = st.x_trans
param_emu['gf_y_trans'] = st.gf_args['y_trans']
param_emu['pl_y_trans'] = st.pl_args['y_trans']
param_emu['qf_y_trans'] = st.qf_args['y_trans']
param_emu['use_mean'] = st.use_mean

root_dir = "./"

assert st.sigma8, "Use sigma8 cosmology for this Emulator interface to Jax-cosmo"
print("Using: Omega_cdm h^2, Omega_b h^2, sigma8, ns, h")

param_emu['load_dir'] = root_dir + '/pknl_components' \
        + st.d_one_plus+"_sig8_" + str(st.nk) + "x" + str(st.nz)

param_emu['kernel_gf']    = kernel_RBF
param_emu['kernel_pklin'] = kernel_RBF
param_emu['kernel_qfunc'] = kernel_Matern12

# -

emu = JemuPk(param_emu)

cosmo_jax = jc.Planck15()

# par = {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 2.9, 'n_s': 1.0, 'h': 0.75}


h_emu = 0.75
omega_c_emu = 0.12 / h_emu**2
omega_b_emu = 0.022 / h_emu**2
sigma8_emu = 0.8
n_s_emu = 1.0
cosmo_jax = jc.Cosmology(Omega_c=omega_c_emu, Omega_b=omega_b_emu, sigma8=sigma8_emu, n_s=n_s_emu, h=h_emu,
                        Omega_k=0., w0=-1.,wa=0.)

Nk=10*st.nk 
k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.k_max_h_by_Mpc,Nk, endpoint=True) #h/Mpc
z_star = jnp.array([0.,1.])

pklin=emu.linear_pk(cosmo_jax,k_star, z_star)

pk_nonlin = emu.nonlinear_pk(cosmo_jax,k_star, z_star)

pklin.shape

# +
plt.figure(figsize=(10,8))

plt.plot(k_star,pklin[0],label=fr"$Pk_{{lin}}$ z={z_star[0]:.2f}")
plt.plot(k_star,pklin[1],label=fr"$Pk_{{lin}}$ z={z_star[1]:.2f}")
plt.plot(k_star,pk_nonlin[0], label=fr"$Pk_{{nl}}$ z={z_star[0]:.2f}")
plt.plot(k_star,pk_nonlin[1], label=fr"$Pk_{{nl}}$ z={z_star[1]:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])
plt.legend();
# -

z_test = 1.0

func = lambda p: emu.linear_pk(p,k_star, z_star=z_test)
func_nl = lambda p: emu.nonlinear_pk(p,k_star, z_star=z_test)

jac_lin_emu = jax.jacfwd(func)(cosmo_jax)
jac_nonlin_emu = jax.jacfwd(func_nl)(cosmo_jax)


# Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa, gamma=None
titles=["Jacobien Pk Lin","Jacobien Pk NonLin"]
fig,ax = plt.subplots(1,2,figsize=(16,8))
for i,jaco in enumerate([jac_lin_emu, jac_nonlin_emu]):
    ax[i].plot(k_star,jaco.h,label="h")
    ax[i].plot(k_star,jaco.Omega_b,label="Omega_b")
    ax[i].plot(k_star,jaco.Omega_c,label="Omeg_c")
    ax[i].plot(k_star,jaco.n_s,label="n_s")
    ax[i].plot(k_star,jaco.w0,label="w0")
    ax[i].plot(k_star,jaco.wa,label="wa")
    ax[i].set_xscale("log")
    ax[i].set_title(titles[i])
    ax[i].legend();

# Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa, gamma=None
Omega_c_arr = jnp.linspace(cosmo_jax.Omega_c*0.5,cosmo_jax.Omega_c*1.5,10)/cosmo_jax.h**2
axes = jc.Cosmology(Omega_c=0,
                    Omega_b=None,h=None,n_s=None,sigma8=None,Omega_k=None,w0=None,wa=None,gamma=None)

pklin_Omegac_emu = jax.vmap(func, in_axes=(axes,))(jc.Cosmology(Omega_c=Omega_c_arr,
                                               Omega_b=cosmo_jax.Omega_b,
                                               h=cosmo_jax.h,
                                               n_s=cosmo_jax.n_s,
                                               sigma8=cosmo_jax.sigma8,
                                               Omega_k=cosmo_jax.Omega_k,
                                               w0=cosmo_jax.w0,
                                               wa=cosmo_jax.wa,
                                               gamma=cosmo_jax.gamma))

pk_nonlin_Omegac_emu = jax.vmap(func_nl, in_axes=(axes,))(jc.Cosmology(Omega_c=Omega_c_arr,
                                               Omega_b=cosmo_jax.Omega_b,
                                               h=cosmo_jax.h,
                                               n_s=cosmo_jax.n_s,
                                               sigma8=cosmo_jax.sigma8,
                                               Omega_k=cosmo_jax.Omega_k,
                                               w0=cosmo_jax.w0,
                                               wa=cosmo_jax.wa,
                                               gamma=cosmo_jax.gamma))

titles=["Pk Lin","Pk NonLin"]
fig,ax = plt.subplots(1,2,figsize=(16,8))
for iax,pk_emu in enumerate([pklin_Omegac_emu, pk_nonlin_Omegac_emu]):
    color = iter(mpl.cm.rainbow(np.linspace(0, 1, pk_emu.shape[0])))
    for i in range(pk_emu.shape[0]):
        c = next(color)
        ax[iax].plot(k_star,pk_emu[i,:],c=c,label=fr"$\Omega_c=${Omega_c_arr[i]:.2f}");
    ax[iax].set_xscale("log")
    ax[iax].set_yscale("log")
    ax[iax].set_title(titles[iax])
    ax[iax].grid()
    ax[iax].set_xlim([1e-3,1e2])
    ax[iax].set_ylim([1e-2,1e6])
    ax[iax].legend();


