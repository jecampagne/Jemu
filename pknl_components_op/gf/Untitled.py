# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: jaxcosmo
#     language: python
#     name: jaxcosmo
# ---

import numpy as np 
# for I/O
import os 

data = None
data = np.load("gp_2.npz", allow_pickle=True)

data.files

data["kernel_hat"].item()['k_length']

data["beta_hat"]


