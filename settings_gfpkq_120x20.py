# Author: J.E Campagne
# Status : Under Development

# -----------------------------------------------------------------------------
# method for building the emulator of pklin(k,z), pknl(k,z)
#

components = True

# if we want to include sample neutrino mass
# this does not imply that neutrino is not included in CLASS
neutrino = False

# sigma8 or As cosmo param
sigma8 = True


if not neutrino:

    # fixed neutrino mass
    fixed_nm = {'M_tot': 0.06}


# minimum redshift
zmin = 0.0

# maximum redshift
zmax = 3.5         # version > 19thJune22 was 4.66

# maximum of k (for quick CLASS run, set to for example, 50)
k_max_h_by_Mpc_TrainingMaker = 5000.

k_max_h_by_Mpc = 50.

# our wanted kmax
kmax = 50.0

# minimum of k
k_min_h_by_Mpc = 5E-4

# number of k
nk = 120

# number of redshift on the grid
nz = 20

# -----------------------------------------------------------------------------

# choose which cosmological parameters to marginalise over
# first 5 are by default

#JEC version>19thJune22 Omega_c & Omega_b instead of omega_c, omega_b

if neutrino:

    # list of cosmological parameters to use
    # we suggest keeping this order since the emulator inputs are in the same order
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
# Baryon Feedback settings

# baryon model to be used
baryon_model = 'AGN'

cst = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
               'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
               'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
       'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
               'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
               'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
       'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                 'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                 'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}


# -----------------------------------------------------------------------------

# Settings for the GP emulator module

# noise/jitter term
var = 1e-5

# another jitter term for numerical stability 
### was 1e-5, but JEC push to 1e-10: which leads to missmatch with oreiginal emuPk results)
jitter = 1e-5   

# order of the polynomial (maximum is 2)
order = 2

# Transform input (pre-whitening)
x_trans = True

# Centre output on 0 if we want
use_mean = False

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


# choice of optimizer (better to use 'L-BFGS-B')
method = 'L-BFGS-B'

# tolerance to stop the optimizer
ftol = 1E-30

# maximum number of iterations
maxiter = 1000

# growth factor (not very broad distribution in function space)
gf_args = {'y_trans': False, 'lambda_cap': 1}

# if we want to emulate 1 + q(k,z):
emu_one_plus_q = True

if emu_one_plus_q:

    # q function (expected to be zero)
    qf_args = {'y_trans': True, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = '_op'

else:

    # q function (expected to be zero)
    qf_args = {'y_trans': False, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = ''

# linear matter power spectrum (at z=0)
pl_args = {'y_trans': True, 'lambda_cap': 1000}

# Scale growth factor
gf_scale_args = {'y_trans': True, 'lambda_cap': 1}

# Q-func tilde (bis)
### sans doute pb qf_bis_args = {'y_trans': True, 'lambda_cap': 1}
qf_bis_args = {'y_trans': True, 'lambda_cap': 1}

# non linear power spectrum (at z=0)
pnl_args = {'y_trans': True, 'lambda_cap': 1000}

