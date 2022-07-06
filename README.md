# Jemu

*Please notice that this repo is under development and most of the materials are to be used with caution.* 

CLASS Pk emulator in JAX
Jump to the [Jemu-demo.ipynb](https://github.com/jecampagne/Jemu/blob/main/Jemu-demo.ipynb) and play with it...
For the time beeing the code is not yet optimized and loading the GPs parameters could take a few minutes. 

Since 6th July 22, I edit some scripts to ease the process to create new CLASS emulator set.
- `make_new_cosmo.py`: build 1000 cosmological parameters in 5D Latin Hypercube. The current version is minimal to regenerate [Omega_cdm, Omega_b, sigma8, n_s,h] dataset.  
  > There are options (needs to read the script fro details)
- `make_trainingset_gfpkq.py`: run CLASS with the cosmological parameter set produced by `make_new_cosmo.py`.

  > There are options (needs to read the script fro details)

  > The running needs to set the parameters of the `settings_default.py` file. 

  > As CLASS can fail we try up to 3 times to generate cosmo set that differ from original by a little shift.
   In case the failure persist then we exit on error. 
  If the new cosmo is ok, then we register it in the cosmo_validated array
  
  The CLASS run can take very long time, so we store the correspondant files at each validation
    of a new cosmology set in a temporary directory indexed by the first cosmology index and the last to date run
    
  One can launch on paralell batch the computation of different ranges of cosmo sets as for instance
  ```python
  python make_trainingset_gfpkq.py --idrange 0, 100    &
  python make_trainingset_gfpkq.py --idrange 100, 200   &
  ```
  
It is the responsability of the user to properly merge the diffrent files afterwards.

