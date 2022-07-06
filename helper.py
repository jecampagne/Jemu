import os

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap, jit
jax.config.update("jax_enable_x64", True)

def load_arrays(folder_name, file_name):

    matrix = jnp.load(folder_name + '/' + file_name + '.npz')['arr_0']

    return matrix

def store_arrays(array, folder_name, file_name):

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # use compressed format to store data
    np.savez_compressed(folder_name + '/' + file_name + '.npz', array)
        
        
