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
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.80'

import numpy as np
import jax
import jax.numpy as jnp
from classy import Class    # CLASS python
import jax_cosmo as jc      # Jax-cosmo lib
import pyccl as ccl         # CCL python
import jemupk  as emu  # Jax Emulator of CLASS
import cosmopower as cp # Cosmo Power (from Alessio Spurio Mancini since 10thJuly22)
# -

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
import matplotlib.patches as mpatches
mpl.rcParams['font.size'] = 20

# +
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total)

nvidia_smi.nvmlShutdown()


# -

# ## Cosmopower Neural Net parameters

# +
cp_dir = "./cosmo_power_trained/"
# instantiate your CP linear power emulator

pklin_cp = cp.cosmopower_NN(restore=True,
                            restore_filename=cp_dir+'/PKLIN_NN') # change with path to your linear power emulator .pkl file, without .pkl suffix


# instantiate your CP nonlinear correction (halofit) emulator 
pknlratio_cp = cp.cosmopower_NN(restore=True,
                                restore_filename=cp_dir+'/PKBOOST_NN') # change with path to your nonlinear correction emulator .pkl file, without .pkl suffix
# -

# ## Jemu parameters

# +
import settings_gfpkq_120x20  as st         # configuration file (update 2/June/22)

print(f"(k,z)-grid: {st.nk}x{st.nz}")

# +
root_dir = "./"
if st.sigma8:
    print("Using: Omega_cdm, Omega_b, sigma8, ns, h")
    tag='_Omega_sig8_'  + str(st.nk) + "x" + str(st.nz)  
else:
    raise NotImplementedError("No more in use")

load_dir = root_dir + '/pknl_components' + st.d_one_plus+tag
# -

# load GP material
gp_factory = emu.GP_factory.make(load_dir)

# ## Define a cosmology parameter set

# + tags=[]
h_emu = 0.6774 
Omega_c_emu = 0.2589
Omega_b_emu = 0.0486
sigma8_emu = 0.8159
n_s_emu = 0.9667

cosmo_ccl_eh = ccl.Cosmology(
    Omega_c=Omega_c_emu, Omega_b=Omega_b_emu, 
    h=h_emu, n_s=n_s_emu, sigma8 = sigma8_emu,
    transfer_function='eisenstein_hu', matter_power_spectrum='halofit')

cosmo_ccl_bc = ccl.Cosmology(
    Omega_c=Omega_c_emu, Omega_b=Omega_b_emu, 
    h=h_emu, n_s=n_s_emu, sigma8 = sigma8_emu,
    transfer_function='boltzmann_class', matter_power_spectrum='halofit')

# -

cosmo_ccl_eh

cosmo_ccl_bc

cosmo_ccl_bc['N_nu_mass'],cosmo_ccl_bc['N_nu_rel']

cosmo_jax = jc.Cosmology(Omega_c=Omega_c_emu, Omega_b=Omega_b_emu, 
    h=h_emu, sigma8=sigma8_emu, n_s=n_s_emu, Omega_k=0.0, w0=-1.0,wa=0.0)

params_cosmo_power = {'Omega_cdm': [Omega_c_emu],
          'Omega_b':   [Omega_b_emu],
          'h':         [h_emu],
          'n_s':       [n_s_emu],
          'sigma8':    [sigma8_emu],
         }

# # Test multi redshifts Pk computations

# Nb. to compare to CosmoPower the k_mode are defined according its definition, but it should be also in the range of Jemu

#Nk=10*st.nk 
#k_star = jnp.geomspace(st.k_min_h_by_Mpc, st.k_max_h_by_Mpc, Nk, endpoint=True) #h/Mpc
k_star = pklin_cp.modes / h_emu

k_star = k_star[k_star>st.k_min_h_by_Mpc]
k_star = k_star[k_star<st.k_max_h_by_Mpc]

z_star = jnp.array([0.,1., 2., 3.])

# ## Compute Pk Lin & Non Lin with Jemu: first call XLA compilation is done, second call the XLA-compiled version is used directly  

# %time pk_linear_interp = emu.linear_pk(cosmo_jax, k_star,z_star)

# %time pk_nonlin_interp = emu.nonlinear_pk(cosmo_jax,k_star, z_star) 

# %timeit pk_linear_interp = emu.linear_pk(cosmo_jax, k_star,z_star)

# %timeit pk_nonlin_interp = emu.nonlinear_pk(cosmo_jax,k_star, z_star) 

pk_linear_interp.shape, pk_nonlin_interp.shape

# +
# CCL & Jax-cosmo

# +
zbin=2
z_ccl = z_star[zbin].item()

print("z_ccl=",z_ccl)
# -

# Prepare Classy

st.fixed_nm['M_tot']

# +
params_def_classy = {
    'output': 'mPk',
    'n_s': n_s_emu, 
    'h': h_emu,
    'Omega_b': Omega_b_emu,
    'Omega_cdm':Omega_c_emu,
    'N_ncdm': 1.0, 
    'deg_ncdm': 3.0, 
    'T_ncdm': 0.71611, 
    'N_ur': 0.00641,
    'z_max_pk' :  st.zmax,
    'P_k_max_h/Mpc' : 500.,
    'halofit_k_per_decade' : 80.,
    'halofit_sigma_precision' : 0.05
    }

if st.sigma8:
    params_def_classy['sigma8']=sigma8_emu
else:
    raise NotImplementedError("No more in use") #params_def_classy['A_s']=As_emu,

#Neutrino default
params_def_classy['m_ncdm'] = st.fixed_nm['M_tot']/params_def_classy['deg_ncdm']

params_classy_nl =  params_def_classy.copy()
params_classy_nl['non_linear'] = 'halofit'
# -


# Clean CLASS memory
try:
    class_module_nl.struct_cleanup()
    class_module_nl.empty()
except:
    pass

class_module_nl = Class()
class_module_nl.set(params_classy_nl)
class_module_nl.compute()


# ## Compute Plin for CLASS, CCL, Jax-cosmo

pk_class_lin = np.array([class_module_nl.pk_lin(k * h_emu, z_ccl) for k in k_star])
pk_lin_ccl_eh = ccl.linear_matter_power(cosmo_ccl_eh, k_star*cosmo_jax.h, 1./(1+z_ccl)) #last is scale factor 1=>z=0
pk_lin_ccl_bc = ccl.linear_matter_power(cosmo_ccl_bc, k_star*cosmo_jax.h, 1./(1+z_ccl)) #last is scale factor 1=>z=0
#pk_lin_jc = jc.power.linear_matter_power(cosmo_jax,k_star, 1./(1+z_ccl))/cosmo_jax.h**3

# ## Compute Pk Non-Lin for CLASS, CCL, Jax-cosmo

# +
pk_class_nl  = np.array([class_module_nl.pk(k * h_emu, z_ccl) for k in k_star])

pk_nonlin_ccl_eh = ccl.nonlin_matter_power(cosmo_ccl_eh, k_star*cosmo_jax.h, 
                                        1./(1+z_ccl)) #last is scale factor 1=>z=0
pk_nonlin_ccl_bc = ccl.nonlin_matter_power(cosmo_ccl_bc, k_star*cosmo_jax.h, 
                                        1./(1+z_ccl)) #last is scale factor 1=>z=0


#pk_nonlin_jc = jc.power.nonlinear_matter_power(cosmo_jax,k_star, 
#                                               1./(1+z_ccl))/cosmo_jax.h**3
# -

# Clean CLASS memory
try:
    class_module_nl.struct_cleanup()
    class_module_nl.empty()
except:
    pass

#  ## Compute Pk lin & Pk Non-Lin for CosmoPower

# +
params_cosmo_power['z'] = [z_ccl]

# get the logarithm of linear matter power spectrum 
log_linear_power = pklin_cp.predictions_np(params_cosmo_power)[0]

pk_cp_lin = 10.**log_linear_power

# get the logarithm of nonlinear correction (halofit)
log_nlratio      = pknlratio_cp.predictions_np(params_cosmo_power)[0]

# get the logarithm of total power.
log_total_power = log_linear_power + log_nlratio

pk_cp_nonlin = 10.**log_total_power

# -

pklin_cp.modes.shape, k_star.shape

x = (pklin_cp.modes/h_emu).copy()
y = k_star.copy()

indices =  np.where(x[:, None] == y[None, :])[1]

# +
plt.figure(figsize=(10,8))

plt.plot(k_star,pk_linear_interp[zbin,:],lw=2, c="b", label="Jemu")
#plt.plot(k_star,pk_lin_jc,lw=2, c="r", label="jax_cosmo")
plt.plot(k_star,pk_lin_ccl_eh,lw=2, ls="--", c="lime", label=r"ccl (EH+Halofit)")
plt.plot(k_star,pk_lin_ccl_bc,lw=2, ls="--", c="green", label=r"ccl (Boltzman+Halofit)")
plt.plot(k_star,pk_class_lin,lw=2, ls=":", c="purple",label="classy")
plt.plot(k_star,pk_cp_lin[indices],lw=2, ls=":", c="cyan",label="CosmoPower")



plt.plot(k_star,pk_nonlin_interp[zbin],lw=2, c="b")#, label=r"$P_{{nl}}$ (Jemu)")
#plt.plot(k_star,pk_nonlin_jc,lw=2, c="r")#, label=r"$P_{{nl}}$ (jax_cosmo)")
plt.plot(k_star,pk_nonlin_ccl_eh,lw=2, ls="--",c="lime")#,label=r"$P_{{nl}}$ (ccl)")
plt.plot(k_star,pk_nonlin_ccl_bc,lw=2, ls="--",c="green")#,label=r"$P_{{nl}}$ (ccl)")
plt.plot(k_star,pk_class_nl,lw=2, ls=":", c="purple")#,label="classy")
plt.plot(k_star,pk_cp_nonlin[indices],lw=2, ls=":", c="cyan")


plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k\ [h\ Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z)\ [Mpc^3]$")
plt.grid()
plt.title(rf"$z={z_ccl:.2f}$");
plt.xlim([1e-3,5e1])
plt.ylim([1e-2,1e6]);

# +
fig, axs= plt.subplots(nrows=2,figsize=(10,8), gridspec_kw={"hspace":0.5})
axs[0].plot(k_star,(pk_linear_interp[zbin,:]-pk_class_lin)/pk_class_lin,lw=2, c="b", label="Jemu")
#axs[0].plot(k_star,(pk_lin_jc-pk_class_lin)/pk_class_lin,lw=2, c="r", label="jax_cosmo (EH + Halofit)")
axs[0].plot(k_star,(pk_lin_ccl_eh-pk_class_lin)/pk_class_lin,lw=2, c="lime", label="ccl (EH + Halofit)")
axs[0].plot(k_star,(pk_lin_ccl_bc-pk_class_lin)/pk_class_lin,lw=2, ls="--", c="lime", label="ccl (Boltzman CLASS + Halofit)")
axs[0].plot(k_star,(pk_cp_lin[indices]-pk_class_lin)/pk_class_lin,lw=2, c="purple", label="CosmoPower")

axs[0].legend(loc='upper left')
axs[0].grid()
axs[0].set_xscale("log");

axs[0].set_ylim([-0.05,0.05])
axs[0].plot([k_star.min(),k_star.max()],[-0.01,-0.01],c='k',ls='--')
axs[0].plot([k_star.min(),k_star.max()],[0.01,0.01],c='k',ls='--')
axs[0].set_xlabel(r"$k\ [h\ Mpc^{-1}]$")

axs[0].set_title(f"Pk Lin: Relative diff. wrt CLASS (z={z_ccl:.2f})");


axs[1].plot(k_star,(pk_nonlin_interp[zbin]-pk_class_nl)/pk_class_nl,lw=2, c="b", label="Jemu")#, label=r"$P_{{nl}}$ (Jemu)")
#axs[1].plot(k_star,(pk_nonlin_jc-pk_class_nl)/pk_class_nl,lw=2, c="r", label="jax-cosmo (EH + Halofit)")#, label=r"$P_{{nl}}$ (jax_cosmo)")
axs[1].plot(k_star,(pk_nonlin_ccl_eh-pk_class_nl)/pk_class_nl,lw=2,c="lime",  label="ccl (EH + Halofit)")#,label=r"$P_{{nl}}$ (ccl)")
axs[1].plot(k_star,(pk_nonlin_ccl_bc-pk_class_nl)/pk_class_nl,lw=2,ls="--", c="lime",  label="ccl (Boltzman CLASS + Halofit)")#,label=r"$P_{{nl}}$ (ccl)")
axs[1].plot(k_star,(pk_cp_nonlin[indices]-pk_class_nl)/pk_class_nl,lw=2,c="purple",  label="CosmoPower")#,label=r"$P_{{nl}}$ (ccl)")

axs[1].grid()
axs[1].legend(loc='upper left')
axs[1].set_xscale("log")
axs[1].set_ylim([-0.05,0.05])
axs[1].plot([k_star.min(),k_star.max()],[-0.01,-0.01],c='k',ls='--')
axs[1].plot([k_star.min(),k_star.max()],[0.01,0.01],c='k',ls='--')
axs[1].set_xlabel(r"$k\ [h\ Mpc^{-1}]$")

axs[1].set_title(f"Pk NLin: Relative diff. wrt CLASS (z={z_ccl:.2f})");


# +
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total)

nvidia_smi.nvmlShutdown()
# -

# # Jacobians & vectorization


# +
#jc_func_nl = lambda p: jc.power.nonlinear_matter_power(p,k_star, 1./(1+z_ccl))/p.h**3
#jac_jc_func_nl = jax.jacfwd(jc_func_nl)(cosmo_jax)
# -

# The fist call triggers the XLA compilation.
func_nl = lambda p: emu.nonlinear_pk(p,k_star, z_star=z_ccl)
jac_nonlin_emu = jax.jacfwd(func_nl)(cosmo_jax)

# ### Notice that the emulator has a fixed (Omega_k, w0, wa) values so the gradients are not relevant for these parameters 

# Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa, gamma=None
titles=f"Jacobien Pk NonLin (color: emu, dashed: jax_cosmo) (z={z_ccl:.2f})"
lines=["-","--"]
colors=[None,"k"]
fig = plt.figure(figsize=(8,8))
for i,jaco in enumerate([jac_nonlin_emu, jac_jc_func_nl]):
    plt.plot(k_star,jaco.h,label="h",c=colors[i],ls=lines[i])
    plt.plot(k_star,jaco.Omega_b,label="Omega_b",c=colors[i],ls=lines[i])
    plt.plot(k_star,jaco.Omega_c,label="Omega_c",c=colors[i],ls=lines[i])
    plt.plot(k_star,jaco.n_s,label="n_s",c=colors[i],ls=lines[i])
    if i == 0:
        plt.legend();
plt.xlabel(r"$k\ [h\ Mpc^{-1}]$")
plt.xscale("log")
plt.title(titles)
plt.grid()

# Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa, gamma=None
Omega_c_arr = jnp.linspace(cosmo_jax.Omega_c*0.5,cosmo_jax.Omega_c*1.5,10)
axes = jc.Cosmology(Omega_c=0,
                    Omega_b=None,h=None,n_s=None,sigma8=None,Omega_k=None,w0=None,wa=None,gamma=None)

pk_nonlin_Omegac_jc = jax.vmap(jc_func_nl, in_axes=(axes,))(
    jc.Cosmology(
        Omega_c=Omega_c_arr,
        Omega_b=cosmo_jax.Omega_b,
        h=cosmo_jax.h,
        n_s=cosmo_jax.n_s,
        sigma8=cosmo_jax.sigma8,
        Omega_k=cosmo_jax.Omega_k,
        w0=cosmo_jax.w0,
        wa=cosmo_jax.wa,
        gamma=cosmo_jax.gamma))

pk_nonlin_Omegac_emu = jax.vmap(func_nl, in_axes=(axes,))(
    jc.Cosmology(
        Omega_c=Omega_c_arr,
        Omega_b=cosmo_jax.Omega_b,
        h=cosmo_jax.h,
        n_s=cosmo_jax.n_s,
        sigma8=cosmo_jax.sigma8,
        Omega_k=cosmo_jax.Omega_k,
        w0=cosmo_jax.w0,
        wa=cosmo_jax.wa,
        gamma=cosmo_jax.gamma))

titles=f"Pk NonLin (color: emu, dashed: jax_cosmo)  (z={z_ccl:.2f})"
fig = plt.figure(figsize=(8,8))
for iax,pk_emu in enumerate([pk_nonlin_Omegac_emu, pk_nonlin_Omegac_jc]):
    color = iter(mpl.cm.rainbow(np.linspace(0, 1, pk_emu.shape[0])))
    for i in range(pk_emu.shape[0]):
        if iax==0:
            c = next(color)
            plt.plot(k_star,pk_emu[i,:],c=c,label=fr"$\Omega_c=${Omega_c_arr[i]:.2f}");
        else:
            if i==0 or i==pk_emu.shape[0]-1:
                plt.plot(k_star,pk_emu[i,:],c='k',ls='--')
plt.xscale("log")
plt.yscale("log")
plt.title(titles)
plt.grid()
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])
plt.legend()
plt.xlabel(r"$k\ [h\ Mpc^{-1}]$");




