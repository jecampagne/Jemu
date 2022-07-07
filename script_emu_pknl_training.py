import numpy as np


from timeit import default_timer as timer

import os
import shutil
import argparse


#######
# Training of Pk Non Linear (k,z=0) 
# It uses
# - the cosmological parameter dataset of make_new_cosmo.py
# - the CLASS Pk file produced by make_trainingset_gfpkq.py
#######



#######
import settings_default as st  #### USER SETTINGS should be the same as for make_trainingset_gfpkq.py
#######


import helper as hp    # I/O

######
# Simple Gaussian process class adapted for the emulator training
from gaussproc_emu_training import *
######

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', default='./', type=str, help='root directory (aka [dir] below) (location of cosmo parameter set and outputs)')
parser.add_argument('--cosmo', type=str, help='cosmology dataset in [dir]/trainingset', required=True)
parser.add_argument('--dump', action='store_true', help='dump first cosmo param. run and do not save')
parser.add_argument('--dirIn', help='directory under [dir]/trainingset containg the CLASS results', required=True)
parser.add_argument('--tagOut', help='tag for directory output: [dir]/pknl_components_<tag>/pnl containing the training results', required=True)



def Job(theta: np.ndarray,
        y: np.ndarray,
        gp_model: GPEmuTraining, 
        gp_args: Dict,
        n_restart: int = 5,
        print_dump: bool = False):
              
        if(print_dump):
            print("Init...")
        gp_model.Init()
        gp_model.mean_Theta_y(theta,y)

        #theta,y => x_train, y_train (using x_trans & y_trans)
        if (print_dump):
            print("do_transformation")
        gp_model.do_transformation()  
        
        # prepare training
        if (print_dump):
            print("prepare_training")
        gp_model.prepare_training()
        
        #do training (optimization of Kernel hyper-parameters
        rng_key = jax.random.PRNGKey(42)
            
        #Todo : switch to jax.lax.while_loop(cond_fun,body,val)
        if (print_dump):
            print("optimize...")
        # first pass
        n_iter=0
        min_funcs = []
        kernel_pars = []
        while n_iter<n_restart :   # test (not rc)
            rng_key, new_key = jax.random.split(rng_key)
            kernel_para, func = gp_model.optimize(new_key, init_param=None)
            #print("param",kernel_para, " func value=",func)
            min_funcs.append(func)
            kernel_pars.append(kernel_para)
            n_iter += 1
        #clean
        min_funcs = jnp.array(min_funcs)
        kernel_pars = jnp.array(kernel_pars)
        if jnp.isnan(min_funcs).any():
            index = jnp.argwhere(jnp.isnan(min_funcs))
            min_funcs    = jnp.delete(min_funcs, index)
            kernel_pars = jnp.delete(kernel_pars, index, axis=0)
        
        
        kernel_para = kernel_pars[min_funcs==jnp.min(min_funcs)][0]
        kernel_para = kernel_para.flatten()
        min_func    = min_funcs[min_funcs==jnp.min(min_funcs)][0]
        if (print_dump):
            print(f"final: k_par: {kernel_para}, func: {min_func}")

                
        #compute qties for predictions
        if (print_dump):
            print("post training...")
        gp_model.set_kernel_hat(kernel_para)
        gp_model.post_training()

        if(print_dump):
            print("final beta_hat: ", gp_model.beta_hat)
        
        #Store gp model
        if(print_dump):
            print("store...")
        gp_model.store_info(gp_args["folder_name"],gp_args["file_name"])

        #end
        if(print_dump):
            print("done...")
        return gp_model


################################
if __name__ == "__main__":

    args = parser.parse_args()


    #############
    root_dir =  args.dir
    dump = args.dump


    if not st.sigma8:
        raise NotImplementedError("Only sigma8 schema is used")

    #############
    # Load cosmological parameter sets
    # Omega_cdm, Omega_b, sigma8, ns, h
    ###########
    dircosmo = root_dir + '/trainingset'
    cosmologies = hp.load_arrays(dircosmo,args.cosmo)
    tag="sigma8"

    if dump:
        print("0)",cosmologies)


    print(f"Cosmo[{tag}]: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")

    print(f"Order of polynomial approx: {st.order}")
    print(f"Whitening of x_train: {st.x_trans}")
    print(f"Transformation of y_tain: {st.gf_args}")
    print(f"outputs are centred on zero: {st.use_mean}")
    print(f"noise covariance matrix: {st.var}")
    print(f"Matrix diag term for stability: { st.jitter}")
    print(f"z range [{st.zmin}, {st.zmax}]")
    print(f"k range [{st.k_min_h_by_Mpc}, {st.k_max_h_by_Mpc_TrainingMaker}]")


    dirName = root_dir + '/trainingset/' + args.dirIn +'/'
    pk_nl = hp.load_arrays(dirName, 'pk_nl')

    print(f"Non Linear Pk: nber of training points {pk_nl.shape[0]} for {pk_nl.shape[1]} k (log)")
    n_pnl = pk_nl.shape[1]
    assert n_pnl == st.nk, "Hummm something strange..."
    print(f"The number of GPs to model Pk NLin={n_pnl} (= nber of k_bins) ")

    folder_pnl = root_dir + '/pknl_components_' + args.tagOut  + '/pnl'

    arg_pnl = [[cosmologies, pk_nl[:, i], st.pnl_args, folder_pnl, 'gp_' + str(i)] for i in range(n_pnl)]


    if dump:
        print("Start Pk Non lin")
    start = timer()


    for i_pnl in range(n_pnl):
	print(f"Process GP_{i_pnl}/{n_pnl}")
	theta = arg_pnl[i_pnl][0] # cosmo \Theta_i i<N_train
	y = arg_pnl[i_pnl][1]     # Pk(kj,\Theta_i, z=0)
	arg_cur_gp = arg_pnl[i_pnl][2]
	arg_cur_gp["folder_name"] = arg_pnl[i_pnl][3]
	arg_cur_gp["file_name"]   = arg_pnl[i_pnl][4]
	
	
	gp_model = GPEmuTraining(kernel=kernel_RBF,
				 var=st.var,
				 order=st.order,
				 lambda_cap=st.pnl_args['lambda_cap'],
				 l_min=st.l_min,
				 l_max=st.l_max,
				 a_min=st.a_min,
				 a_max=st.a_max,
				 jitter=st.jitter,
				 x_trans=st.x_trans,
				 y_trans=st.pnl_args['y_trans'],
				 use_mean=st.use_mean)
	
	
	gf_job = Job(theta=theta, y=y, 
		     gp_model=gp_model,
		     gp_args=arg_cur_gp,             
		     n_restart= st.n_restart,
		     print_dump=True
		     )



    end = timer()
    print(f"end-start (sec): {end - start}")
    print('All Done')

