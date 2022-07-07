import sys
import shutil
import argparse

from timeit import default_timer as timer

import numpy as np
from classy import Class
from classy import CosmoComputationError

import helper as hp      # load/save 


###
#   CLASS RUN with the cosmological parameter set
#   as it can fail with try up to 3 times to generate cosmo set that differ from original
#   by a little shift.
#   In case the failure persist then with exit
#   If the new cosmo is ok, then we register it in the cosmo_validated array
###
#
# The CLASS run can take very long time, so we store the correspondant files at each validation
#    of a new cosmology set
#
# version >19thJune22
# Use case [Omega_cdm, Omega_b, sigma8, ns, h] cosmmo. No more "As" parameter 
# Run CLASS with minimal neutrino settings.


#######
import settings_default as st     #### USER SETTINGS 
#######


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', default='./', type=str, help='root directory (aka [dir] below) (location of cosmo parameter set and outputs)')
parser.add_argument('--cosmo', type=str, help='cosmology dataset in [dir]/trainingset', required=True)
parser.add_argument('--dump', action='store_true', help='dump first cosmo param. run and do not save')
parser.add_argument('--dirOut',default="components_new", help='final directory under [dir]/trainingset containing the CLASS results (we add the ifrst & last cosmo index processed.')
parser.add_argument('--idrange', type=int, nargs=2, default=[0,1000], help='cosmo parameter index in the dataset [first, last)')



##########################
# Generate training set for GaussProc optim.
# case: cosmo param. with sigma8 and Growth/Pklin/Qfunc computation
##########################
def make_class_dict(cosmo, sigma8=True):
    """
    CLASS parameters dictionnary
    """
    Omega_c_emu = cosmo[0]    
    Omega_b_emu = cosmo[1]
    n_s_emu = cosmo[3]
    h_emu = cosmo[4]
    
    class_dict_def ={
        'output': 'mPk',
        'n_s': n_s_emu, 
        'h': h_emu,
        'Omega_b': Omega_b_emu,
        'Omega_cdm':Omega_c_emu,
        'N_ncdm': 1.0, 
        'deg_ncdm': 3.0, 
        'T_ncdm': 0.71611, 
        'N_ur': 0.00641,
        'z_max_pk' : st.zmax,
        'P_k_max_h/Mpc' : st.k_max_h_by_Mpc_TrainingMaker,
        'halofit_k_per_decade' : 80.,
        'halofit_sigma_precision' : 0.05
    }
    
    if sigma8:
        sigma8_emu =  cosmo[2]    #sigma8
        class_dict_def['sigma8'] = sigma8_emu
    else:
        ln1010As_emu = cosmo[2]   # Ln(10^10 As)
        As_emu = 10**(-10)*np.exp(ln1010As_emu)
        class_dict_def['A_s']= As_emu


    #Neutrino default
    class_dict_def['m_ncdm'] = st.fixed_nm['M_tot']/class_dict_def['deg_ncdm']

    class_dict_nl = class_dict_def.copy()
    class_dict_nl['non_linear'] = 'halofit'
    
    return class_dict_nl

################################
if __name__ == "__main__":

    args = parser.parse_args()


    #############
    root_dir =  args.dir

    dump = args.dump
    if dump:
        print("################")
        print("################")
        print("## DUMP MODE  ##")
        print("################")
        print("################")


    if st.sigma8:
        dircosmo = root_dir + '/trainingset'
        cosmologies = hp.load_arrays(dircosmo,args.cosmo)
        tag="sigma8"
        if dump:
            print("0)",cosmologies)
    else:
        raise NotImplementedError("As cosmo: No more in use")

    print(f"Cosmo[{tag}]: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")


    ifirst_cosmo = args.idrange[0]
    ilast_cosmo  = args.idrange[1]
    assert ilast_cosmo <= cosmologies.shape[0], f"idrange max value ={ilast_cosmo} > nber of cosmo param.[{cosmologies.shape[0]}] "

    if dump:
        print(f"cosmo index:[{ifirst_cosmo}, {ilast_cosmo}]")

    all_pklin0  = []
    all_pknl0         = []
    all_growth_kscale = []
    all_qfunc_bis     = []
    

    # 2D (k,z) grid for Pknl(k,z) but 1D for D(z) and Pklin(k,z=zref) 
    k_g = np.geomspace(st.k_min_h_by_Mpc,st.k_max_h_by_Mpc,st.nk, endpoint=True) #h/Mpc
    z_g = np.linspace(st.zmin,st.zmax,st.nz, endpoint=True)
    zref = 0.0


    #Loop on cosmology param
    print("Start....")
    start = timer()
    cosmo_validated =[]
    for ic in range(ifirst_cosmo,ilast_cosmo):

        cosmo = cosmologies[ic]
        
        if dump and ic>1:
            break
            
        tries = 3
        for it in range(tries):
            try: 
                print(f"Cosmo[{ic}]:",cosmo)
                start_cur = timer()

                params_nl = make_class_dict(cosmo)

                #Prepare CLASS for Non Linear Pk
                class_module_nl = Class()
                class_module_nl.set(params_nl)
                class_module_nl.compute()
            
            except CosmoComputationError:
                if it < tries - 1:
                    print(f"failure [{it}] update cosmo")
                    cosmo += 1e-4 * np.random.randn(cosmologies.shape[1])
                    continue
                else:
                    raise
            break

        
        cosmo_validated.append(cosmo)

        # Compute Pk linear at zref Plin(k,zref) and compute the Growth D(z) = P(kref,z)/P(kref,zref) , zref=0
        pklin0 = np.array([class_module_nl.pk_lin(k_g[k] * params_nl['h'], zref) for k in range(st.nk)])

        # Define D(k,z) =  P(k,z)/P(k,zref)      new 8/6/22 
        pklin_grid = class_module_nl.get_pk_all(k_g*params_nl['h'],z_g, nonlinear=False)
        # ouput has shape (st.nz, st.nk)
        # => transform into (st.nk,st.nz)
        pklin_grid  = pklin_grid.T 

        growth_kscale = pklin_grid/pklin0.reshape(st.nk,1)

        if dump:
            print("2b)",growth_kscale.shape)


        ##############
        # Pk Non lin
        ##############
        pknl0 = np.array([class_module_nl.pk(k_g[k] * params_nl['h'], zref) for k in range(st.nk)]) # Pk_nl(k,zref=0)
        
        pknl  = class_module_nl.get_pk_all(k_g*params_nl['h'],z_g)  # Pk_nl(k,z)
        # Pknl ouput has shape (st.nz, st.nk)
        # => transform into (st.nk,st.nz)
        pknl  = pknl.T

        if dump:
            print("3)",pknl.shape, pknl0.shape)


        #Qfunc_bis(k,z) =  Pk_nl(k,z) /  Pk_nl(k,z=0) - 1
        q_func_bis = pknl / pknl0.reshape(st.nk,1) - 1

        if dump:
            print("4)",q_func_bis.shape)
        
        
        #store
        all_pklin0.append(pklin0.flatten())
        all_growth_kscale.append(growth_kscale.flatten())
        all_pknl0.append(pknl0.flatten())
        all_qfunc_bis.append(q_func_bis.flatten())

        
        #temporary storage

        
        dirName =  root_dir + 'trainingset/component_tmp'+ str(ifirst_cosmo)+ "_" + str(ic) + "/" 
        if dump:
            print("Dump: SAve temporary files in : ", dirName)
        fn_pklin = 'pk_linear'
        fn_pknl = 'pk_nl'
        fn_growth_kscale = 'growth_factor_kscale'
        fn_q_func_bis = 'q_function_bis'

        hp.store_arrays(all_pklin0,  dirName, fn_pklin)
        hp.store_arrays(all_growth_kscale, dirName, fn_growth_kscale)
        hp.store_arrays(all_pknl0, dirName, fn_pknl)
        hp.store_arrays(all_qfunc_bis, dirName, fn_q_func_bis)
        hp.store_arrays(cosmo_validated,dirName, 'cosmo_validated')

        # remove previous intermediate storage
        if ic>0:
            old_ic = ic-1
            old_dir = root_dir + 'trainingset/component_tmp'+ str(ifirst_cosmo)+ "_" + str(old_ic) + "/"
            shutil.rmtree(old_dir, ignore_errors=True)
        
        
        # Clean CLASS memory
        class_module_nl.struct_cleanup()
        class_module_nl.empty()

        ##
        end_cur = timer()
        print(f"Time cur(sec): {end_cur - start_cur}")
        

    ##########
    ### Save
    ##########
    print("Save....")

    dirName =  root_dir + '/trainingset/' + args.dirOut+ '_' + \
              str(ifirst_cosmo)+ "_" + str(ic) + "/"
        
    if dump:
        print("Dump:  files in : ", dirName)

    fn_pklin = 'pk_linear'
    fn_pknl = 'pk_nl'
    fn_growth_kscale = 'growth_factor_kscale'
    fn_q_func_bis = 'q_function_bis'
    
    if st.neutrino:
       fn_pklin  = fn_pklin  + "_neutrino"
       fn_pknl = fn_pknl + "_neutrino"
       fn_growth_kscale = fn_growth_kscale + "_neutrino"
       fn_q_func_bis = fn_q_func_bis + "_neutrino"
       

    print(f"Save in directory {dirName}")

    if not dump:
        hp.store_arrays(all_pklin0,  dirName, fn_pklin)
        hp.store_arrays(all_growth_kscale, dirName, fn_growth_kscale)
        hp.store_arrays(all_pknl0, dirName, fn_pknl)
        hp.store_arrays(all_qfunc_bis, dirName, fn_q_func_bis)
        hp.store_arrays(cosmo_validated,dirName, 'cosmo_validated')

    #
    end = timer()
    print(f"All done (sec): {end - start}")
    print("END please clean the directry: ",root_dir + '/trainingset')
