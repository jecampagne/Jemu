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

import numpy as np
from classy import Class

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 20

h_emu = 0.73022459
omega_c_emu = 0.12616742    # omega_c h^2
omega_b_emu = 0.02163407   #omega_b h^2
n_s_emu = 1.15724975
ln1010As_emu = 2.72607173
As_emu = 10**(-10)*np.exp(ln1010As_emu)

As_emu

#omega_x = Omega_x h^2
#
params_def_nl = {
    'output': 'mPk',
    'A_s': As_emu,
    'n_s': n_s_emu, 
    'h': h_emu,
    'omega_b': omega_b_emu,
    'omega_cdm':omega_c_emu,
    'N_ncdm': 1.0, 
    'deg_ncdm': 3.0, 
    'T_ncdm': 0.71611, 
    'N_ur': 0.00641,
    'm_ncdm':0.02,
    'non_linear' : 'halofit',
    'z_max_pk' : 4.66,
    'P_k_max_h/Mpc' : 50.,                  #No h-unit
    'halofit_k_per_decade' : 80.,
    'halofit_sigma_precision' : 0.05
    }


# +
class_module = Class()

class_module.set(params_def_nl)

class_module.compute()
# -

class_module.sigma8()

# +
k_g = np.geomspace(5e-4,50,80) #h/Mpc
z_g = np.linspace(0.,4.66,20)

pknl_g = class_module.get_pk_all(k_g*h_emu,z_g) 



z_g
# -

pknl_g.shape

Nk = 400
k_star = np.geomspace(5e-4, 5e1,Nk, endpoint=True)  #h/Mpc
z_star = z_g[5]
pk_matter = np.array([class_module.pk(k * h_emu, z_star) for k in k_star])


plt.figure(figsize=(10,8))
plt.plot(k_star, pk_matter, c="b")
plt.plot(k_g,pknl_g[5,:],c='r',ls='--')
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k [h Mpc^{-1}]$")
plt.ylabel(r"$P_\delta(k,z) [Mpc^3]$")
plt.grid()
plt.xlim([1e-3,1e2])
plt.ylim([1e-2,1e6])
plt.legend();

# # Test de generation de nouveau trainingset

from helper import *              # load/save   

root_dir = "./"

cosmologies = load_arrays(root_dir + 'trainingset/components', 'cosmologies')
print(f"Cosmo: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")

cosmologies[0]


def make_class_dict(cosmo):

    omega_c_emu = cosmo[0]    # omega_c h^2
    omega_b_emu = cosmo[1]   #omega_b h^2
    ln1010As_emu = cosmo[2]
    As_emu = 10**(-10)*np.exp(ln1010As_emu)
    n_s_emu = cosmo[3]
    h_emu = cosmo[4]
    
    class_dict_def ={
        'output': 'mPk',
        'A_s': As_emu,
        'n_s': n_s_emu, 
        'h': h_emu,
        'omega_b': omega_b_emu,
        'omega_cdm':omega_c_emu,
        'N_ncdm': 1.0, 
        'deg_ncdm': 3.0, 
        'T_ncdm': 0.71611, 
        'N_ur': 0.00641,
        'm_ncdm':0.02,
        'z_max_pk' : 4.66,
        'P_k_max_h/Mpc' : 50.,                 
        'halofit_k_per_decade' : 80.,
        'halofit_sigma_precision' : 0.05
    }
    class_dict_lin = class_dict_def.copy()
    class_dict_lin['non_linear'] = 'none'
    class_dict_nl = class_dict_def.copy()
    class_dict_nl['non_linear'] = 'halofit'
    
    return class_dict_lin, class_dict_nl


params_lin, params_nl = make_class_dict(cosmologies[0])

# +
class_module_lin = Class()
class_module_lin.set(params_lin)
class_module_lin.compute()

class_module_nl = Class()
class_module_nl.set(params_nl)
class_module_nl.compute()

# +
k_g = np.geomspace(5e-4,50,80, endpoint=True) #h/Mpc
z_g = np.linspace(0.,4.66,20, endpoint=True)

pklin = class_module_lin.get_pk_all(k_g*params_lin['h'],z_g) 
pknl = class_module_nl.get_pk_all(k_g*params_nl['h'],z_g) 
# -

type(z_g[5]), type(pklin[0,0])

pklin.shape, pknl.shape

pklin_z0 = pklin[0,:]

pklin_z0.shape

growth0 = pklin[:,0]/pklin[0,0]

growth0.shape

growth_grid = pklin/pklin[0,:]

growth_grid.shape

mean_growth_grid = np.mean(growth_grid, axis=1)
std_growth_grid = np.std(growth_grid, axis=1)

plt.plot(z_g,growth0)
plt.errorbar(z_g, mean_growth_grid,yerr=std_growth_grid, fmt='o')

plt.errorbar(z_g, growth0-mean_growth_grid,yerr=std_growth_grid, fmt='o')
plt.xlabel("z")
plt.title(r"$D(z,k_0)-\langle D(z,k)\rangle_k$");

qfunc = pknl/pklin -1.0

qfunc.shape

np.dot( growth.reshape(growth.shape[0],1), pklin_z0.reshape(1,pklin_z0.shape[0])).shape

tmp1=(qfunc+1.0)*np.dot( growth.reshape(growth.shape[0],1), pklin_z0.reshape(1,pklin_z0.shape[0]))

tmp = (qfunc+1.0)*np.dot(pklin_z0.reshape(pklin_z0.shape[0],1), growth.reshape(1,growth.shape[0]))

tmp1-pknl

plt.hist(((tmp1-pknl)/pknl).flatten())


(qfunc +1)*pklin - pknl

# # Transformation of As->sigma8 cosmologies

root_dir = "./"

# Original set {Omega_cdm h^2, Omega_b h^2, ln(10^10 As), ns, h}
cosmologies_Aslike = load_arrays(root_dir + 'trainingset', 'cosmologies')
print(f"Cosmo: nber of training Cosmo points {cosmologies_Aslike.shape[0]} for {cosmologies_Aslike.shape[1]} params")

cosmologies_Aslike.shape

#cosmologies_sig8like = cosmologies_Aslike.copy()
sig8_lin = []
sig8_nl = []
for ic, cosmo in enumerate(cosmologies_Aslike):
    if ic%10 == 0:
        print(f"Cosmo[{ic}]...")
    params_lin, params_nl = make_class_dict(cosmo)
    class_module_lin = Class()
    class_module_lin.set(params_lin)
    class_module_lin.compute()
    sig8_lin.append(class_module_lin.sigma8())
    
    class_module_nl = Class()
    class_module_nl.set(params_nl)
    class_module_nl.compute()
    sig8_nl.append(class_module_nl.sigma8())

    # Clean CLASS memory
    class_module_lin.struct_cleanup()
    class_module_lin.empty()
    class_module_nl.struct_cleanup()
    class_module_nl.empty()


sig8_lin_tmp = np.array(sig8_lin)
sig8_nl_tmp  = np.array(sig8_nl)

np.max(np.abs(sig8_lin_tmp-sig8_nl_tmp))

cosmologies_Aslike.shape

ln1010As = cosmologies_Aslike[:,2]

ln1010As.min(), ln1010As.max()

plt.scatter(ln1010As,sig8_nl, s=2)
plt.xlabel(r"$\ln 10^{10} A_s$")
plt.ylabel(r"$\sigma_8$")

plt.hist(sig8_nl,bins=50);
plt.xlabel("sigma8")
plt.ylabel("counts");

sig8_nl_tmp.min(), sig8_nl_tmp.max()

plt.hist(ln1010As,bins=50);

plt.scatter(cosmologies_Aslike[:,0],sig8_nl, s=2)

plt.scatter(cosmologies_Aslike[:,1],sig8_nl, s=2)

plt.scatter(cosmologies_Aslike[:,3],sig8_nl, s=2)

plt.scatter(cosmologies_Aslike[:,4],sig8_nl, s=2)

import arviz as az
import corner

# +
# #! pip install arviz

# +
# #! pip install corner
# -

data={}
data['ln1010As']=cosmologies_Aslike[:,2]
data['sigma8']=sig8_nl_tmp
data['Omega_c']=cosmologies_Aslike[:,0]
data['Omega_b']=cosmologies_Aslike[:,1]
data['n_s']=cosmologies_Aslike[:,3]
data['h']=cosmologies_Aslike[:,4]

import arviz.labels as azl
from matplotlib.colors import ListedColormap
cmap1 = ListedColormap(['blue'])
labeller = azl.MapLabeller(var_name_map={"Omega_c": r"$\Omega_c$", 
                                             "sigma8": r"$\sigma_8$",
                                             "h":r"$h$", "Omega_b": r"$\Omega_b$"})

axes1= az.plot_pair(
        data,
        kind="kde",
        labeller=labeller,
#        var_names=var_nm,
        marginal_kwargs={"plot_kwargs": {"lw":2, "c":'b', "ls":"-"}},
        kde_kwargs={
            "hdi_probs": [0.3],  # Plot 68% and 90% HDI contours
            #"hdi_probs":[0.393, 0.865, 0.989],  # 1, 2 and 3 sigma contours
            "contour_kwargs":{"colors":None, "cmap":cmap1, "linewidths":2,
                              "linestyles":"-"},
            "contourf_kwargs":{"alpha":1},
        },
        point_estimate_kwargs={"lw": 3, "c": "b"},
        marginals=True, textsize=35, point_estimate='median',
    );


cosmologies_sigma8 = cosmologies_Aslike.copy()

cosmologies_sigma8[:,2] = sig8_lin_tmp

pwd

np.save(root_dir+"/trainingset/"+"cosmologies_sig8.npz",cosmologies_sigma8)

# # Generate cosmologies from Latin Hypercube 5D

import scipy

dist=scipy.stats.uniform(loc=1,scale=2)

dist.ppf(0.1)


def scale(t,loc,scale):
    return loc + t * (scale)


scale(0.1,1,2)

import pandas as pd

pwd

latHypSpl = pd.read_csv('/sps/lsst/users/campagne/emuPK/emulator/'+ 'lhs/' + 'maximin_1000_5D', index_col=0).values

latHypSpl.shape

# +
# Uniform(loc,scale) => uniform dist in [loc, loc+scale]
# -

(0.25, 4.-0.25)

priors = {
    'omega_cdm': {'distribution': 'uniform', 'specs': [0.06, 0.34]},    # [0.06, 0.40]
    'omega_b':   {'distribution': 'uniform', 'specs': [0.019, 0.007]},  # [0.019, 0.026]
    'sigma8':    {'distribution': 'uniform', 'specs': [0.25, 3.0]},     # [0.25, 3.25]
    'n_s':       {'distribution': 'uniform', 'specs': [0.70, 0.60]},    # [0.7, 1.3]
    'h':         {'distribution': 'uniform', 'specs': [0.64, 0.18]}     # [0.64, 0.82]
}


new_cosmo = np.zeros_like(latHypSpl)

tmp = priors['omega_cdm']

eval('scipy.stats.'+tmp['distribution'])(*tmp['specs'])

# +
stats = {}

for c in priors:
    tmp = priors[c]
    stats[c] = eval('scipy.stats.'+tmp['distribution'])(*tmp['specs'])
# -

stats

new_cosmo = np.zeros_like(latHypSpl)
for i,p in enumerate(priors):
    new_cosmo[:,i] = stats[p].ppf(latHypSpl[:,i])

new_cosmo

plt.hist(new_cosmo[:,2]);

np.savez(root_dir+"/trainingset/"+"cosmologies_sig8.npz",new_cosmo)

cosmologies[0]

params_lin, params_nl = make_class_dict(cosmologies[0])

class_module_lin = Class()
class_module_lin.set(params_lin)
class_module_lin.compute()

k_g = np.geomspace(5e-4,50,40, endpoint=True) #h/Mpc
z_g = np.linspace(0.,4.66,20, endpoint=True)

pklin_all=class_module_lin.get_pk_all(k_g*params_lin['h'],z_g)

pklin_all.shape

plt.plot(k_g,pklin_all[0,:])
plt.yscale("log")
plt.xscale("log")

pklin0=pklin_all[0,:]

plt.plot(pklin_all[:,0]/pklin0[0])

pklin_all[:,0]/pklin0[0]

pklin0_bis = np.array([class_module_lin.pk_lin(k_g[k] * params_lin['h'], 0.0) for k in range(40)])


pklin0_bis/pklin0

# # Dev > 16 June 22

from helper import *   
import jax_cosmo as jc

import scipy.interpolate as itp

root_dir = "./"

# {Omega_cdm h^2, Omega_b h^2, sigma8, ns, h}
cosmologies = load_arrays(root_dir + 'trainingset/components_sig8_120x20', 'cosmo_validated')
print(f"Cosmo: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")


def emu2jc(theta):
    omega_c_emu, omega_b_emu, sigma8_emu, n_s_emu, h_emu = theta 
    
    return jc.Cosmology(Omega_c=omega_c_emu/h_emu**2, 
                 Omega_b=omega_b_emu/h_emu**2, 
                 h=h_emu, 
                 n_s=n_s_emu,
                 sigma8=sigma8_emu,   
                 Omega_k=0.,
                 w0=-1.0,
                 wa=0.0)


omega_c_arr = []
omega_b_arr = []
Omega_c_arr = []
Omega_b_arr = []
h_arr = []
sigma8_arr = []
Omega_m_arr = []   # secondary param
Omega_de_arr = []  #   "
ns_arr = []
for ic, cosmo in enumerate(cosmologies):
    if ic%100 == 0:
        print(f"Cosmo[{ic}]...")
    cosmo_jc = emu2jc(cosmo)
    omega_c_arr.append(cosmo_jc.Omega_c * cosmo_jc.h**2)
    omega_b_arr.append(cosmo_jc.Omega_b * cosmo_jc.h**2)
    Omega_c_arr.append(cosmo_jc.Omega_c)
    Omega_b_arr.append(cosmo_jc.Omega_b)
    h_arr.append(cosmo_jc.h)
    ns_arr.append(cosmo_jc.n_s)
    sigma8_arr.append(cosmo_jc.sigma8)
    Omega_m_arr.append(cosmo_jc.Omega_m)
    Omega_de_arr.append(cosmo_jc.Omega_de)


# +
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 14
fig,axs=plt.subplots(2,5,figsize=(17,10), gridspec_kw={"wspace":0.35})

axs[0,0].hist(omega_c_arr, bins=50, density=True);
axs[0,0].set_xlabel("$\omega_c$");
axs[0,1].hist(omega_b_arr, bins=50, density=True);
axs[0,1].set_xlabel("$\omega_b$");
axs[0,2].hist(sigma8_arr, bins=50, density=True);
axs[0,2].set_xlabel("$\sigma_8$");
axs[0,3].hist(h_arr, bins=50, density=True);
axs[0,3].set_xlabel("$h$");
axs[0,4].hist(ns_arr, bins=50, density=True);
axs[0,4].set_xlabel("$n_s$");

axs[1,0].hist(Omega_c_arr, bins=50, density=True);
axs[1,0].set_xlabel("$\Omega_c$");
axs[1,1].hist(Omega_b_arr, bins=50, density=True);
axs[1,1].set_xlabel("$\Omega_b$");
axs[1,2].hist(Omega_m_arr, bins=50, density=True);
axs[1,2].set_xlabel("$\Omega_m$");
axs[1,3].hist(Omega_de_arr, bins=50, density=True);
axs[1,3].set_xlabel("$\Omega_{de}$");

fig.delaxes(axs[1,4])

# -

def make_class_dict(cosmo):

    omega_c_emu = cosmo[0]    # omega_c h^2
    omega_b_emu = cosmo[1]   #omega_b h^2
    sigma8_emu  = cosmo[2]
    n_s_emu = cosmo[3]
    h_emu = cosmo[4]
    
    class_dict_def ={
        'output': 'mPk',
        'sigma8': sigma8_emu,
        'n_s': n_s_emu, 
        'h': h_emu,
        'omega_b': omega_b_emu,
        'omega_cdm':omega_c_emu,
        'N_ncdm': 1.0, 
        'deg_ncdm': 3.0, 
        'T_ncdm': 0.71611, 
        'N_ur': 0.00641,
        'm_ncdm':0.02,
        'z_max_pk' : 4.66,
        'P_k_max_h/Mpc' : 50.,                 
        'halofit_k_per_decade' : 80.,
        'halofit_sigma_precision' : 0.05
    }
    class_dict_lin = class_dict_def.copy()
    class_dict_lin['non_linear'] = 'none'
    class_dict_nl = class_dict_def.copy()
    class_dict_nl['non_linear'] = 'halofit'
    
    return class_dict_lin, class_dict_nl


As_arr = []
k_pivot = 0.05 # default CLASS 
# As := Exp[ln(P_primodial(k=k_pivot))]
for ic, cosmo in enumerate(cosmologies):
    if ic%100 == 0:
        print(f"Cosmo[{ic}]...")

    params_lin, params_nl = make_class_dict(cosmo)
    class_module_lin = Class()
    class_module_lin.set(params_lin)
    class_module_lin.compute()
    
    pk_primo = class_module_lin.get_primordial()  #
    spline = itp.splrep(pk_primo['k [1/Mpc]'], pk_primo['P_scalar(k)'])
    As = itp.splev(k_pivot, spline).item()*1e9
    As_arr.append(As)
    
    class_module_lin.struct_cleanup()
    class_module_lin.empty()


def plot_loghist(x, bins, ax):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  ax.hist(x, bins=logbins)
  ax.set_xscale('log')



# +
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 14
fig,axs=plt.subplots(2,5,figsize=(17,10), gridspec_kw={"wspace":0.35})

axs[0,0].hist(omega_c_arr, bins=50, density=True);
axs[0,0].set_xlabel("$\omega_c$");
axs[0,1].hist(omega_b_arr, bins=50, density=True);
axs[0,1].set_xlabel("$\omega_b$");
axs[0,2].hist(sigma8_arr, bins=50, density=True);
axs[0,2].set_xlabel("$\sigma_8$");
axs[0,3].hist(h_arr, bins=50, density=True);
axs[0,3].set_xlabel("$h$");
axs[0,4].hist(ns_arr, bins=50, density=True);
axs[0,4].set_xlabel("$n_s$");

axs[1,0].hist(Omega_c_arr, bins=50, density=True);
axs[1,0].set_xlabel("$\Omega_c$");
axs[1,1].hist(Omega_b_arr, bins=50, density=True);
axs[1,1].set_xlabel("$\Omega_b$");
axs[1,2].hist(Omega_m_arr, bins=50, density=True);
axs[1,2].set_xlabel("$\Omega_m$");
axs[1,3].hist(Omega_de_arr, bins=50, density=True);
axs[1,3].set_xlabel("$\Omega_{de}$");
#axs[1,4].hist(As_arr, bins=50, density=True);

hist, bins = np.histogram(As_arr, bins=50)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
axs[1,4].hist(As_arr, bins=logbins)
axs[1,4].set_xlabel("$10^{-9}A_s$");
axs[1,4].set_xscale("log")


# -
# # Try DES Y3 


import pandas as pd
import scipy

latHypSpl = pd.read_csv('/sps/lsst/users/campagne/emuPK/emulator/'+ 'lhs/' + 'maximin_1000_5D', index_col=0).values

priors = {
    'Omega_cdm': {'distribution': 'uniform', 'specs': [0.1, 0.4]},    # [0.1, 0.5]
    'Omega_b':   {'distribution': 'uniform', 'specs': [0.04, 0.02]},  # [0.04, 0.06]
    'sigma8':    {'distribution': 'uniform', 'specs': [0.7, 0.2]},     # [0.7, 0.9]
    'n_s':       {'distribution': 'uniform', 'specs': [0.87, 0.19]},    # [0.87, 1.06]
    'h':         {'distribution': 'uniform', 'specs': [0.55, 0.25]}     # [0.55, 0.8]
}

priors.keys()

stats = {}
for c in priors:
    tmp = priors[c]
    stats[c] = eval('scipy.stats.'+tmp['distribution'])(*tmp['specs'])

new_cosmo = np.zeros_like(latHypSpl)
for i,p in enumerate(priors):
    new_cosmo[:,i] = stats[p].ppf(latHypSpl[:,i])

new_cosmo

np.argmax(new_cosmo[:,1] * (new_cosmo[:,4]**2))

new_cosmo[863]


def make_class_dict(cosmo):

    Omega_c_emu = cosmo[0]
    Omega_b_emu = cosmo[1]
    #As_emu  = cosmo[2] * 10**(-9)
    sigma8_emu = cosmo[2]
    n_s_emu = cosmo[3]
    h_emu = cosmo[4]
    
    class_dict_def ={
        'output': 'mPk',
        'sigma8': sigma8_emu,
        'n_s': n_s_emu, 
        'h': h_emu,
        'Omega_b': Omega_b_emu,              #### Omega and not omega=Omega h^2
        'Omega_cdm': Omega_c_emu,            #### idem
        'N_ncdm': 1.0, 
        'deg_ncdm': 3.0, 
        'T_ncdm': 0.71611, 
        'N_ur': 0.00641,
        'm_ncdm':0.02,
        'z_max_pk' : 4.66,
        'P_k_max_h/Mpc' : 50.,                 
        'halofit_k_per_decade' : 80.,
        'halofit_sigma_precision' : 0.05
    }
    class_dict_lin = class_dict_def.copy()
    class_dict_lin['non_linear'] = 'none'
    
    return class_dict_lin


As_arr = []
k_pivot = 0.05 # default CLASS 
# As := Exp[ln(P_primodial(k=k_pivot))]
for ic, cosmo in enumerate(cosmologies):
    if ic%100 == 0:
        print(f"Cosmo[{ic}]...")

    params_lin = make_class_dict(cosmo)
    class_module_lin = Class()
    class_module_lin.set(params_lin)
    class_module_lin.compute()
    
    pk_primo = class_module_lin.get_primordial()  #
    spline = itp.splrep(pk_primo['k [1/Mpc]'], pk_primo['P_scalar(k)'])
    As = itp.splev(k_pivot, spline).item()*1e9
    As_arr.append(As)
    
    class_module_lin.struct_cleanup()
    class_module_lin.empty()

"""
sig8_arr = []
for ic, cosmo in enumerate(new_cosmo):
    if ic%100 == 0:
        print(f"Cosmo[{ic}]...")

    params_lin  = make_class_dict(cosmo)
    #print(f"omega_b[{ic}]=",params_lin['Omega_b']*params_lin['h']**2)
    
    class_module_lin = Class()
    class_module_lin.set(params_lin)
    class_module_lin.compute()
    
    sig8 = class_module_lin.sigma8()
    sig8_arr.append(sig8)
    
    class_module_lin.struct_cleanup()
    class_module_lin.empty()
"""


def emu2jc(theta):
    Omega_c_emu, Omega_b_emu, sigma8_emu, n_s_emu, h_emu = theta 
    
    return jc.Cosmology(Omega_c=Omega_c_emu, 
                 Omega_b=Omega_b_emu, 
                 h=h_emu, 
                 n_s=n_s_emu,
                 sigma8=sigma8_emu,   
                 Omega_k=0.,
                 w0=-1.0,
                 wa=0.0)


omega_c_arr = []
omega_b_arr = []
Omega_c_arr = []
Omega_b_arr = []
h_arr = []
sigma8_arr = []
Omega_m_arr = []
Omega_de_arr = []
ns_arr = []
for ic, cosmo in enumerate(new_cosmo):
    if ic%100 == 0:
        print(f"Cosmo[{ic}]...")
#####    cosmo[2] = sig8_arr[ic]         ######## 
    cosmo_jc = emu2jc(cosmo)
    omega_c_arr.append(cosmo_jc.Omega_c * cosmo_jc.h**2)
    omega_b_arr.append(cosmo_jc.Omega_b * cosmo_jc.h**2)
    Omega_c_arr.append(cosmo_jc.Omega_c)
    Omega_b_arr.append(cosmo_jc.Omega_b)
    h_arr.append(cosmo_jc.h)
    ns_arr.append(cosmo_jc.n_s)
    sigma8_arr.append(cosmo_jc.sigma8)
    Omega_m_arr.append(cosmo_jc.Omega_m)
    Omega_de_arr.append(cosmo_jc.Omega_de)

# +
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 14
fig,axs=plt.subplots(2,5,figsize=(17,10), gridspec_kw={"wspace":0.35})

n,_,_=axs[0,0].hist(Omega_c_arr, bins=50, density=True);
axs[0,0].set_xlabel("$\Omega_c$");
omega_c_fidu=0.12
h_fidu=0.67
Omega_c_fidu = omega_c_fidu/h_fidu**2
axs[0,0].plot([Omega_c_fidu,Omega_c_fidu],[0,np.max(n)],c='r',ls="--",lw=3)

n,_,_=axs[0,1].hist(Omega_b_arr, bins=50, density=True);
axs[0,1].set_xlabel("$\Omega_b$");
omega_b_fidu=0.022
Omega_b_fidu = omega_b_fidu/h_fidu**2
axs[0,1].plot([Omega_b_fidu,Omega_b_fidu],[0,np.max(n)],c='r',ls="--",lw=3)

n,_,_=axs[0,2].hist(sigma8_arr, bins=50, density=True);
axs[0,2].set_xlabel("$\sigma_8$");
sigma8_fidu=0.81
axs[0,2].plot([sigma8_fidu,sigma8_fidu],[0,np.max(n)],c='r',ls="--",lw=3)


n,_,_=axs[0,3].hist(h_arr, bins=50, density=True);
axs[0,3].set_xlabel("$h$");
axs[0,3].plot([h_fidu,h_fidu],[0,np.max(n)],c='r',ls="--",lw=3)

n,_,_=axs[0,4].hist(ns_arr, bins=50, density=True);
axs[0,4].set_xlabel("$n_s$");
ns_fidu = 0.96
axs[0,4].plot([ns_fidu,ns_fidu],[0,np.max(n)],c='r',ls="--",lw=3)


axs[1,0].hist(omega_c_arr, bins=50, density=True);
axs[1,0].set_xlabel("$\omega_c$");
axs[1,1].hist(omega_b_arr, bins=50, density=True);
axs[1,1].set_xlabel("$\omega_b$");

axs[1,2].hist(Omega_m_arr, bins=50, density=True);
axs[1,2].set_xlabel("$\Omega_m$");
axs[1,3].hist(Omega_de_arr, bins=50, density=True);
axs[1,3].set_xlabel("$\Omega_{de}$");

##axs[1,4].hist(As_arr, bins=50, density=True);
#axs[1,4].hist(new_cosmo[:,2], bins=50, density=True)
##axs[1,4].set_xlabel("$10^{-9}A_s$");
##axs[1,4].set_xscale("log")

hist, bins = np.histogram(As_arr, bins=50, density=True)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
axs[1,4].hist(As_arr, bins=logbins)
axs[1,4].set_xlabel("$10^{-9}A_s$");
axs[1,4].set_xscale("log")
# -
new_cosmo


np.savez(root_dir+"/trainingset/"+"cosmologies_Omega_sig8.npz",new_cosmo)


