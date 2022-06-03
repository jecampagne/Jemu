from timeit import default_timer as timer

import numpy as np
from classy import Class

import helper as hp      # load/save 
import settings_gfpkq as st

##########################
# Generate training set for GaussProc optim.
# case: cosmo param. with sigma8 and Growth/Pklin/Qfunc computation
##########################
def make_class_dict(cosmo, sigma8=True):
    """
    CLASS parameters dictionnary
    """

    omega_c_emu = cosmo[0]    # omega_c h^2
    omega_b_emu = cosmo[1]    #omega_b h^2
    n_s_emu = cosmo[3]
    h_emu = cosmo[4]
    
    class_dict_def ={
        'output': 'mPk',
        'n_s': n_s_emu, 
        'h': h_emu,
        'omega_b': omega_b_emu,
        'omega_cdm':omega_c_emu,
        'N_ncdm': 1.0, 
        'deg_ncdm': 3.0, 
        'T_ncdm': 0.71611, 
        'N_ur': 0.00641,
        'z_max_pk' : st.zmax,
        'P_k_max_h/Mpc' : st.k_max_h_by_Mpc,                 
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


    class_dict_lin = class_dict_def.copy()
    class_dict_lin['non_linear'] = 'none'
    class_dict_nl = class_dict_def.copy()
    class_dict_nl['non_linear'] = 'halofit'
    
    return class_dict_lin, class_dict_nl

################################
if __name__ == "__main__":

    #############

    root_dir = "./"

    dump = False  ### debug ####
    if dump:
        print("################")
        print("################")
        print("##### DUMP #####")
        print("################")
        print("################")

    if st.sigma8:
        #############
        # Load cosmological parameter sets
        # Omega_cdm h^2, Omega_b h^2, sigma8, ns, h
        ###########
        cosmologies = hp.load_arrays(root_dir + 'trainingset','cosmologies_sig8')
        tag="sigma8"

        if dump:
            print("0)",cosmologies)
    else:
        #############
        # Load cosmological parameter sets
        # Omega_cdm h^2, Omega_b h^2, ln10^10As, ns, h
        ###########
        cosmologies = hp.load_arrays(root_dir + 'trainingset', 'cosmologies_As')
        tag="As"

    print(f"Cosmo[{tag}]: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")
        

    all_pklin  = []
    all_growth = []
    all_qfunc  = []

    # 2D (k,z) grid for Pknl(k,z) but 1D for D(z) and Pklin(k,z=zref) 
    k_g = np.geomspace(st.k_min_h_by_Mpc,st.k_max_h_by_Mpc,st.nk, endpoint=True) #h/Mpc
    z_g = np.linspace(st.zmin,st.zmax,st.nz, endpoint=True)
    zref = 0.0


    #Loop on cosmology param
    print("Start....")
    start = timer()
    for ic, cosmo in enumerate(cosmologies):

        print(f"Cosmo[{ic}]:",cosmo)
        start_cur = timer()
        
        params_lin, params_nl = make_class_dict(cosmo)

        #Prepare CLASS for Linear Pk
        class_module_lin = Class()
        class_module_lin.set(params_lin)
        class_module_lin.compute()

        #Prepare CLASS for Non Linear Pk
        class_module_nl = Class()
        class_module_nl.set(params_nl)
        class_module_nl.compute()



        # Compute Pk linear at zref Plin(k,zref) and compute the Growth = P(kref,z)/P(kref,zref)


        pklin0 = np.array([class_module_lin.pk_lin(k_g[k] * params_lin['h'], zref) for k in range(st.nk)])
        pklin  = np.array([class_module_lin.pk_lin(k_g[0] * params_lin['h'],z_g[z]) for z in range(st.nz)])


        if dump:
            print("1)",pklin.shape, pklin0.shape)
        

        growth = pklin/pklin0[0]

        if dump:
            print("2)",growth)

        # Pk Non lin
        pknl  = class_module_nl.get_pk_all(k_g*params_nl['h'],z_g)
        # Pknl ouput has shape (st.nz, st.nk)
        # => transform into (st.nk,st.nz)
        # and flatten before storage
        pknl  = pknl.T

        if dump:
            print("3)",pknl.shape)

        # Qfunc
        q_func = pknl/(np.dot(pklin0.reshape(st.nk,1), growth.reshape(1,st.nz))) - 1.0

        if dump:
            print("4)",q_func.shape,"\n",q_func[:5,:10])
        
        
        #store
        all_pklin.append(pklin0.flatten())
        all_growth.append(growth.flatten())
        all_qfunc.append(q_func.flatten())

        # Clean CLASS memory
        class_module_lin.struct_cleanup()
        class_module_lin.empty()
        class_module_nl.struct_cleanup()
        class_module_nl.empty()

        ##
        end_cur = timer()
        print(f"Time cur(sec): {end_cur - start_cur}")
        

    ##########
    ### Save
    ##########
    print("Save....")

    if st.sigma8:
        dirName =  root_dir + 'trainingset/components_sig8/'
    else:
        dirName =  root_dir + 'trainingset/components_As/'
        
    fn_pklin = 'pk_linear'
    fn_growth= 'growth_factor'
    fn_q_func= 'q_function'

    
    if st.neutrino:
       fn_pklin  = fn_pklin  + "_neutrino"
       fn_growth = fn_growth + "_neutrino"
       fn_q_func = fn_q_func + "_neutrino"


    print(f"Save in directory {dirName}, files {fn_pklin}/{fn_growth}/{fn_q_func}")

    if not dump:
        hp.store_arrays(all_pklin,  dirName, fn_pklin)
        hp.store_arrays(all_growth, dirName, fn_growth)
        hp.store_arrays(all_qfunc,  dirName, fn_q_func)

    
    #
    end = timer()
    print(f"All done (sec): {end - start}")
