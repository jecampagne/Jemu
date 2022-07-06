# Author: J.E Campagne
# Status : Under Development

# -----------------------------------------------------------------------------
# method for building the emulator of pklin(k,z), pknl(k,z)
#

components = True    #### DO NOT MODIFY 

# if we want to include sample neutrino mass
# this does not imply that neutrino is not included in CLASS
neutrino = False   

# sigma8 or As cosmo param
sigma8 = True  #### DO NOT MODIFY  


if not neutrino:

    # fixed neutrino mass
    fixed_nm = {'M_tot': 0.06}


# minimum redshift
zmin = 0.0

# maximum redshift
zmax = 3.5         # version > 19thJune22 was 4.66

# maximum of k (for quick CLASS run, set to for example, 50)
k_max_h_by_Mpc_TrainingMaker = 50.  # use 5000 for stable perf. but quite long run

k_max_h_by_Mpc = 50.

# minimum of k
k_min_h_by_Mpc = 5E-4

# number of k
nk = 120

# number of redshift on the grid
nz = 20

# -----------------------------------------------------------------------------

#JEC version>19thJune22 Omega_c & Omega_b instead of omega_c, omega_b
#     As parameter no more in use (here just for a reminder of old version)

if neutrino:

    # list of cosmological parameters to use
    # KEEP this order since the emulator inputs are in the same order
    if sigma8:
        cosmology = ['Omega_cdm', 'Omega_b', 'sigma8', 'n_s', 'h', 'M_tot']
    else:
        cosmology = ['Omega_cdm', 'Omega_b', 'ln10^{10}A_s', 'n_s', 'h', 'M_tot']

else:
    if sigma8:
        cosmology = ['Omega_cdm', 'Omega_b', 'sigma8', 'n_s', 'h']
    else:
        cosmology = ['Omega_cdm', 'Omega_b', 'ln10^{10}A_s', 'n_s', 'h']

# -----------------------------------------------------------------------------

# Settings for the GP emulator module

# noise/jitter term
var = 1e-5    ### DO NOT MODIFY

# another jitter term for numerical stability 
### was 1e-5, but JEC push to 1e-10: which leads to missmatch with oreiginal emuPk results)
jitter = 1e-5   ### DO NOT MODIFY

# order of the polynomial (maximum is 2)
order = 2   ### DO NOT MODIFY

# Transform input (pre-whitening)
x_trans = True   ### DO NOT MODIFY

# Centre output on 0 if we want
use_mean = False ### DO NOT MODIFY

# Number of times we want to restart the optimiser
n_restart = 5

# minimum lengthscale (in log) -5 -> 0
l_min = -5.0    # was -5.0

# maximum lengthscale (in log)
l_max = 5.0    # was 5.0

# minimum amplitude (in log) was 0.0 -> -5.0 =>exp(-5)\sim 7e-3
a_min = -5.0   # was -5.0 

# maximum amplitude (in log) 25 ->10  (5 for Growth at least)
a_max = 5.0 # was 5.0


emu_one_plus_q = True ## DO NOT MODIFY

if emu_one_plus_q:

    # folder where we will store the files
    d_one_plus = '_op'

else:

    # folder where we will store the files
    d_one_plus = ''

# linear matter power spectrum (at z=0)
pl_args = {'y_trans': True, 'lambda_cap': 1000}       #### KEEP 'y_trans': True

# Scale growth factor
gf_scale_args = {'y_trans': True, 'lambda_cap': 1}  #### KEEP 'y_trans': True

# Q-func tilde (bis)
qf_bis_args = {'y_trans': True, 'lambda_cap': 1}      #### KEEP 'y_trans': True

# non linear power spectrum (at z=0)
pnl_args = {'y_trans': True, 'lambda_cap': 1000}     #### KEEP 'y_trans': True

