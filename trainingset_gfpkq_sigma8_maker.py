from timeit import default_timer as timer

import numpy as np
from classy import Class
from classy import CosmoComputationError    ########## 

import helper as hp      # load/save 
import settings_gfpkq_120x20 as st

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
        'P_k_max_h/Mpc' : st.k_max_h_by_Mpc_TrainingMaker,  #  was st.k_max_h_by_Mpc (paper set to 5000 to get smooth qfunc)
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
        

    all_pklin0  = []
    all_growth  = []
    all_qfunc   = []

    all_pknl0         = []   # new 8/6/22
    all_growth_kscale = []   # new 8/6/22
    all_qfunc_bis     = []   # new 8/6/22
    

    # 2D (k,z) grid for Pknl(k,z) but 1D for D(z) and Pklin(k,z=zref) 
    k_g = np.geomspace(st.k_min_h_by_Mpc,st.k_max_h_by_Mpc,st.nk, endpoint=True) #h/Mpc
    z_g = np.linspace(st.zmin,st.zmax,st.nz, endpoint=True)
    zref = 0.0


    #Loop on cosmology param
    print("Start....")
    start = timer()
    cosmo_validated =[]
    for ic in range(cosmologies.shape[0]):

        cosmo = cosmologies[ic]
        
        if dump:
            assert ic==0, "End of dump"

        tries = 3
        for it in range(tries):
            try: 
                print(f"Cosmo[{ic}]:",cosmo)
                start_cur = timer()

                params_lin, params_nl = make_class_dict(cosmo)

                #Prepare CLASS for Linear Pk
        #        class_module_lin = Class()
        #        class_module_lin.set(params_lin)
        #        class_module_lin.compute()

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

#        # Compute Pk linear at zref Plin(k,zref) and compute the Growth D(z) = P(kref,z)/P(kref,zref) , zref=0


#        pklin0 = np.array([class_module_lin.pk_lin(k_g[k] * params_lin['h'], zref) for k in range(st.nk)])
#        pklin  = np.array([class_module_lin.pk_lin(k_g[0] * params_lin['h'],z_g[z]) for z in range(st.nz)])


#        if dump:
#            print("1)",pklin.shape, pklin0.shape)
        

#        growth = pklin/pklin0[0]

#        if dump:
#            print("2)",growth.shape)


#       # Define D(k,z) =  P(k,z)/P(k,zref)      new 8/6/22 
#        pklin_grid = class_module_nl.get_pk_all(k_g*params_nl['h'],z_g, nonlinear=False)
#        # ouput has shape (st.nz, st.nk)
#        # => transform into (st.nk,st.nz)
#        pklin_grid  = pklin_grid.T 

#        growth_kscale = pklin_grid/pklin0.reshape(st.nk,1)

#        if dump:
#            print("2b)",growth_kscale.shape)


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

#        # Qfunc(k,z) = Pk_nl(k,z) / Pk_lin(k,z=0) x D(z)  - 1       # original definition
#        q_func = pknl/(np.dot(pklin0.reshape(st.nk,1), growth.reshape(1,st.nz))) - 1.0


        #Qfunc_bis(k,z) =  Pk_nl(k,z) /  Pk_nl(k,z=0) - 1    new 8/6/22
        q_func_bis = pknl / pknl0.reshape(st.nk,1) - 1

        if dump:
            print("4)",q_func.shape,q_func_bis.shape)
        
        
        #store
#        all_pklin0.append(pklin0.flatten())
#        all_growth.append(growth.flatten())
#        all_qfunc.append(q_func.flatten())
#        all_growth_kscale.append(growth_kscale.flatten())  # new 8/6/22
        all_pknl0.append(pknl0.flatten())                  # new 8/6/22
        all_qfunc_bis.append(q_func_bis.flatten())         # new 8/6/22

        # Clean CLASS memory
#        class_module_lin.struct_cleanup()
#        class_module_lin.empty()
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
        dirName =  root_dir + 'trainingset/components_sig8_'+ str(st.nk) + "x" + str(st.nz) +'/'
    else:
        dirName =  root_dir + 'trainingset/components_As/'
        
    fn_pklin = 'pk_linear'
    fn_growth= 'growth_factor'
    fn_q_func= 'q_function'
    fn_pknl = 'pk_nl'                           # new 8/6/22
    fn_growth_kscale = 'growth_factor_kscale'   # new 8/6/22
    fn_q_func_bis = 'q_function_bis'            # new 8/6/22
    
    if st.neutrino:
       fn_pklin  = fn_pklin  + "_neutrino"
       fn_growth = fn_growth + "_neutrino"
       fn_q_func = fn_q_func + "_neutrino"
       fn_pknl = fn_pknl + "_neutrino"                      # new 8/6/22
       fn_growth_kscale = fn_growth_kscale + "_neutrino"    # new 8/6/22
       fn_q_func_bis = fn_q_func_bis + "_neutrino"          # new 8/6/22
       

    print(f"Save in directory {dirName}, files {fn_pklin}/{fn_growth}/{fn_q_func}/{fn_pknl}/{fn_growth_kscale}/{fn_q_func_bis}")

    if not dump:
#        hp.store_arrays(all_pklin0,  dirName, fn_pklin)
#        hp.store_arrays(all_growth, dirName, fn_growth)
#        hp.store_arrays(all_qfunc,  dirName, fn_q_func)
#        hp.store_arrays(all_growth_kscale, dirName, fn_growth_kscale) # new 8/6/22
        hp.store_arrays(all_pknl0, dirName, fn_pknl)                  # new 8/6/22
        hp.store_arrays(all_qfunc_bis, dirName, fn_q_func_bis)        # new 8/6/22
        hp.store_arrays(cosmo_validated,dirName, 'cosmo_validated')   # new 12/6/22
    
    #
    end = timer()
    print(f"All done (sec): {end - start}")
