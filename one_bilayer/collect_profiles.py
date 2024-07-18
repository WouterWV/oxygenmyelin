#!/usr/bin/env python 

import numpy as np
from outreading import read_F_D_edges, read_many_profiles
from utils import init_rate_matrix
from analyzeprofiles import extend_profile_water, extend_profile

def get_pbc_rate_matrix(F,D):
    n = len(F)
    rate = init_rate_matrix(n, F, D, pbc=True)
    return rate

def get_rate_matrix(F,D):
    n = len(F)
    rate = init_rate_matrix(n,F,D,pbc=True)
    # undo PBC
    rate[0,-1] = 0.
    rate[-1,0] = 0.
    # Constant A B BCs
    rate[0,0] = rate[0,1] = 0
    rate[-1,-1] = rate[-1,-2] = 0

    return rate

def propagate_with_R(time,R):
    prop = scipy.linalg.expm(time*R)
    init = np.zeros(len(prop))
    init[0] = 1.
    profile = np.dot(prop,init)
    return profile

def diagonalize_rate_matrix(R):
    eigval, eigvec = np.linalg.eig(R)
    # Sort from low to high, such that eigvec[:,-1] and eigvec[:,-2] are the lambda=0 eigvectors
    idx_sort = np.flip(eigval.argsort()[::-1])
    eigval = eigval[idx_sort]
    eigvec = eigvec[:,idx_sort]
    Diag = np.diag(eigval)
    Q = eigvec # R = Q.D.Q^-1, and thus exp(Rt) = Q.exp(Dt).Q^-1
    Qinv = np.linalg.inv(Q)
    max_diff = np.max(np.abs(np.matmul(np.matmul(Q,Diag),np.linalg.inv(Q))-R))
    print("||R - Q.D.Q^-1||_max = "+str(max_diff))
    return Diag,Q,Qinv

def propagate_with_diagonal(time,Diag,Q,Qinv):
    Diag_exp = np.zeros_like(Diag)
    for i in range(len(Diag)):
        Diag_exp[i,i] = np.exp(Diag[i,i]*time)
    prop = np.matmul(np.matmul(Q,Diag_exp),Qinv)
    init = np.zeros(len(prop))
    init[0]=1.
    profile = np.dot(prop,init)
    return profile

def get_eigenrates(R):
    eigval, eigvec = np.linalg.eig(R)
    # Sort from low to high, such that eigvec[:,-1] and eigvec[:,-2] are the lambda=0 eigvectors
    idx_sort = np.flip(eigval.argsort()[::-1])
    eigval = eigval[idx_sort]
    eigvec = eigvec[:,idx_sort]
    return eigval, eigvec
    
# Cutting the membrane part out of F and D profiles #
#####################################################

fn = "fit.popc.select.dat"
F,D,edges = read_F_D_edges(fn)
Drad = D # Not used, but just such that it has same dimensions.

nbins = len(F)
zpbc = edges[-1] - edges[0]
dz = (edges[-1]-edges[0])/(len(edges)-1.) #angstrom
dt = 1.

# BILAYER = bin [12,...,87] = 76 bins
# add 1 bin resembling water to the left and the right
# so 1 bin water + bilayer + 1 bin water = [11,...,88] = 78 bins
# And that is what we select, and define as 'short':
st = 11
end = 88

v_short = F[st:end+1]
d_short = D[st:end+1]
drad_short = Drad[st:end+1]
edges_short = edges[st:end+2] # has an extra element

profile_list = []
R_list = []

nwat = 10
multlist = [11,12,13,14,15]

all_profiles = []
Sinfs=[]
RCs=[]

for mult in multlist:

    expectedlen = mult*len(v_short)+(mult-1)*nwat
    st1 = 0
    end1 = expectedlen - 1

    print("mult:            ",mult)
    print("nwat:            ",nwat)
    print("v_short:         ",len(v_short))
    print("expected length: ",expectedlen, " = mult*len(v_short) + (mult-1)*nwat")

    # Adding water to the short profile
    v_ext, d_ext, drad_ext, edges_ext = extend_profile_water(v_short,d_short,drad_short,edges_short,nwat)
    assert len(v_ext) == len(v_short)+nwat

    # Multiply this water_membrane profile
    vmult,dmult,dradmult,edgesmult = extend_profile(v_ext,d_ext,drad_ext,edges_ext,mult)

    # Remove extra water on the right
    vmult = vmult[:-nwat]
    dmult = dmult[:-nwat]
    dradmult = dradmult[:-nwat]
    edgesmult = edgesmult[:-nwat]
    
    R = get_rate_matrix(vmult,dmult)
    Diag,Q,Qinv = diagonalize_rate_matrix(R)
    evals, evecs = get_eigenrates(R)
    S_inf = np.sum(evecs[:,-1]/evecs[-1,-1])*dz
    
    profiles = []
    times = []

    for time in np.arange(0.1,1,0.01):
        times.append(time)
    for time in np.arange(1,10,0.1):
        times.append(time)
    for time in np.arange(10,100,1):
        times.append(time)
    for time in np.arange(100,1000,10):
        times.append(time)
    for time in np.arange(1000,10000,100):
        times.append(time)
    for time in np.arange(10000,100000,1000):
        times.append(time)
    for time in np.arange(100000,1000000,10000):
        times.append(time)
    for time in np.arange(1000000,10000000,100000):
        times.append(time)
    for time in np.arange(10000000,100000000,1000000):
        times.append(time)

    print(len(times))
    for i,time in enumerate(times):
        profile = propagate_with_diagonal(time,Diag,Q,Qinv)
        profiles.append(profile)
        if (i+1)%int(len(times)/10)==0:
            print("did 10%")
    np.save("profiles_nwat_"+str(nwat)+"_mult_"+str(mult)+".npy",np.array(profiles))
    np.save("times_nwat_"+str(nwat)+"_mult_"+str(mult)+".npy",np.array(times))
    np.save("R_nwat_"+str(nwat)+"_mult_"+str(mult)+".npy",R)
