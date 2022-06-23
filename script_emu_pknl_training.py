
import numpy as np


from timeit import default_timer as timer

import os
import logging

import settings_gfpkq_120x20  as st         # configuration file (update 2/June/22)
import helper as hp   
# Simple Gaussian process class adapted for the emulator
from gaussproc_emu_training import *

#######
# Training of Pk Non Linear (k,z=0) 
#######


def get_logger(name: str, log_name: str, folder_name: str = 'logs'):
    '''
    Create a log file for each Python scrip
    :param: name (str) - name of the Python script
    :param: log_name (str) - name of the output log file
    '''
    # create the folder if it does not exist
    if not os.path.exists(folder_name+'/'+log_name):
        os.makedirs(folder_name+'/'+log_name)


    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'


    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename=folder_name + '/' + log_name + '.log',
                        filemode='w')


    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)


    return logging.getLogger(name)

logger = get_logger('training', 'script_training_DPlinOneQ', 'logs')


# # Training of Pklin
root_dir = "./"
dump = True


if st.sigma8:
    #############
    # Load cosmological parameter sets
    # Omega_cdm, Omega_b, sigma8, ns, h
    ###########
    dircosmo = root_dir + 'trainingset'
    cosmologies = hp.load_arrays(dircosmo,'cosmologies_Omega_sig8')

    tag="sigma8"


    if dump:
        print("0)",cosmologies)
else:
    raise NotImplementedError("As cosmo: No more in use")



print(f"Cosmo[{tag}]: nber of training Cosmo points {cosmologies.shape[0]} for {cosmologies.shape[1]} params")

print(f"Order of polynomial approx: {st.order}")
print(f"Whitening of x_train: {st.x_trans}")
print(f"Transformation of y_tain: {st.gf_args}")
print(f"outputs are centred on zero: {st.use_mean}")
print(f"noise covariance matrix: {st.var}")
print(f"Matrix diag term for stability: { st.jitter}")
print(f"z range [{st.zmin}, {st.zmax}]")
print(f"k range [{st.k_max_h_by_Mpc}, {st.k_min_h_by_Mpc}]")


#########
if st.sigma8:
    dirName = root_dir + 'trainingset/components_Omega_sig8_'+ str(st.nk) + "x" + str(st.nz) +'/'
    pk_nl = hp.load_arrays(dirName, 'pk_nl')
else:
    raise NotImplementedError("not yet avaliable")
    
print(f"Non Linear Pk: nber of training points {pk_nl.shape[0]} for {pk_nl.shape[1]} k (log)")
n_pnl = pk_nl.shape[1]
assert n_pnl == st.nk, "Hummm something strange..."
print(f"The number of GPs to model Pk NLin={n_pnl} (= nber of k_bins) ")

if  st.sigma8:
    folder_pnl = root_dir + '/pknl_components' + st.d_one_plus +'_Omega_sig8_' + str(st.nk) + "x" + str(st.nz) +  '_RBF' + '/pnl'
else:
    raise NotImplementedError("not yet avaliable")
    
arg_pnl = [[cosmologies, pk_nl[:, i], st.pnl_args, folder_pnl, 'gp_' + str(i)] for i in range(n_pnl)]



def Job(theta: np.ndarray,
        y: np.ndarray,
        gp_model: GPEmuTraining, 
        gp_args: Dict,
        n_restart: int = 5,
        print_dump: bool = False):
              
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

                
        #compute qties for predictions (care: jit fnt cannot use self.<var> = ...) (leaks)
        if (print_dump):
            print("post training...")
        gp_model.set_kernel_hat(kernel_para)
        gp_model.post_training()
        
        #Store gp model: Todo do a cleaning before
        print("store...")
        gp_model.store_info(gp_args["folder_name"],gp_args["file_name"])

        #end
        print("Done")
        return gp_model

# +
logger.info("Start Pklin")
start = timer()

for i_pnl in range(n_pnl):
    print(f"Process GP_{i_pnl}/{n_pnl}")
    theta = arg_pnl[i_pnl][0] # cosmo \Theta_i i<N_train
    y = arg_pnl[i_pnl][1]     # Pk(kj,\Theta_i, z=0)
    arg_cur_gp = arg_pnl[i_pnl][2]
    arg_cur_gp["folder_name"] = arg_pnl[i_pnl][3]
    arg_cur_gp["file_name"] = arg_pnl[i_pnl][4]
 

    # GP emulator should be done each time due to jit
    # Todo: see how to change GPEmu to avoid
    gp_model = GPEmuTraining(kernel=kernel_RBF,          # was kernel_RBF,
                         var=st.var,
                         order=st.order,
                         lambda_cap=st.pnl_args['lambda_cap'],   #####ICI
                         l_min=st.l_min,
                         l_max=st.l_max,
                         a_min=st.a_min,
                         a_max=st.a_max,
                         jitter=st.jitter,
                         x_trans=st.x_trans,
                         y_trans=st.pnl_args['y_trans'],         ####ICI
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
