import os
import scipy
from scipy import stats
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', default='./', type=str, help='root directory (location of lhs dir and outputs)')
parser.add_argument('--fn',default="new_cosmo", help='file name of the new cosmology parameter dataset')
parser.add_argument('--Ocdm', type=float,  default=[0.1,0.5], nargs=2, help='Omega_cdm min, max values')
parser.add_argument('--Ob', type=float,  default=[0.04,0.06], nargs=2, help='Omega_baryon min, max values')
parser.add_argument('--sig8', type=float,  default=[0.7,0.9], nargs=2, help='sigma8 min, max values')
parser.add_argument('--ns', type=float,  default=[0.87,1.06], nargs=2, help='n_s min, max values')
parser.add_argument('--h', type=float,  default=[0.55,0.80], nargs=2, help='h min, max values')


def main():

    args = parser.parse_args()

    #Lattin Hypercube points : 1000 pts in 5D space
    latHypSpl = pd.read_csv(args.dir+ 'lhs/' + 'maximin_1000_5D', index_col=0).values

    Omega_cdm_min = args.Ocdm[0]
    Omega_cdm_max = args.Ocdm[1]
    Omega_cdm_delta = Omega_cdm_max - Omega_cdm_min
    assert Omega_cdm_delta>0, "not valid range for -Ocdm option"

    Omega_b_min = args.Ob[0]
    Omega_b_max = args.Ob[1]
    Omega_b_delta = Omega_b_max - Omega_b_min
    assert Omega_b_delta>0, "not valid range for -Ob option"


    sig8_min = args.sig8[0]
    sig8_max = args.sig8[1]
    sig8_delta = sig8_max - sig8_min
    assert sig8_delta>0, "not valid range for -Ocdm option"


    ns_min = args.ns[0]
    ns_max = args.ns[1]
    ns_delta = ns_max - ns_min
    assert ns_delta>0, "not valid range for -ns option"

    h_min = args.h[0]
    h_max = args.h[1]
    h_delta = h_max - h_min
    assert h_delta>0, "not valid range for -h option"


    # Prepare the generation of the cosmo parameter set using scipy distribution
    priors = {
        'Omega_cdm': {'distribution': 'uniform', 'specs': [Omega_cdm_min, Omega_cdm_delta]},
        'Omega_b':   {'distribution': 'uniform', 'specs': [Omega_b_min, Omega_b_delta]},
        'sigma8':    {'distribution': 'uniform', 'specs': [sig8_min, sig8_delta]},
        'n_s':       {'distribution': 'uniform', 'specs': [ns_min, ns_delta]},
        'h':         {'distribution': 'uniform', 'specs': [h_min, h_delta]}
        }

    stats = {}
    for c in priors:
        tmp = priors[c]
        stats[c] = eval('scipy.stats.'+tmp['distribution'])(*tmp['specs'])


    # Generation du dataset
    new_cosmo = np.zeros_like(latHypSpl)
    for i,p in enumerate(priors):
        new_cosmo[:,i] = stats[p].ppf(latHypSpl[:,i])


    # Save the dataset
    fname = args.dir+"/trainingset/"+args.fn+".npz"

    assert not os.path.exists(fname), f"the output ({fname}) file exists..."
            
    print("saving in <",fname,">")
    np.savez(fname, new_cosmo)

if __name__ == '__main__':
    main()
