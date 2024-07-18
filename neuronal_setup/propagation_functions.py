import numpy as np
import scipy 

### We use some functions from mcdiff: ###
def read_F_D_edges(filename):
    """File contains F and D in the following format
    bin-number  start-of-bin  end-of-bin  F(bin)  D(bin-to-next-bin)
    Lines starting with # are skipped.
    Assume F in units [kBT] and D in units [angstrom**2/ps]."""

    with open(filename,"r") as f:
        F = []
        D = []
        bins_str = []   # beginning of bin
        bins_end = []   # end of bin
    
        startline = "   index  bin-str  bin-end"
        for line in f:
            if line.startswith(startline): 
                break
        for line in f:
            if line.startswith("="):
                break
            if not line.startswith("#"):
                words = line.split()
                bins_str.append(float(words[1]))
                bins_end.append(float(words[2]))
                F.append(float(words[3]))
                if len(words) >= 5:
                    D.append(float(words[4]))
                elif len(words)!=4 and len(words)!=6:
                    raise ValueError("error in the format line"+line)
    edges = bins_str+[bins_end[-1]]  # last bin edge is added
    # this returns three vectors: F values, D values, edges values
    return np.array(F),np.array(D),np.array(edges)

def init_rate_matrix_pbc(n,v,w):
    """
    from annekegh mcdiff:
    initialize rate matrix from potential vector v and diffusion
    vector w = log(D(i)/delta^2)"""
    assert len(v) == n  # number of bins
    assert len(w) == n
    rate = np.float64(np.zeros((n,n)))  # high precision

    # off-diagonal elements
    diffv = v[1:]-v[:-1] #length n-1  # diffv[i] = v[i+1]-v[i]
    exp1 = w[:n-1]-0.5*diffv
    exp2 = w[:n-1]+0.5*diffv
    rate.ravel()[n::n+1] = np.exp(exp1)[:n-1]
    rate.ravel()[1::n+1] = np.exp(exp2)[:n-1]
    #this amounts to doing:
    #for i in range(n-1):
    #    rate[i+1,i] = np.exp(w[i]-0.5*(v[i+1]-v[i]))
    #    rate[i,i+1] = np.exp(w[i]-0.5*(v[i]-v[i+1]))

    # corners    # periodic boundary conditions
    rate[0,-1]  = np.exp(w[-1]-0.5*(v[0]-v[-1]))
    rate[-1,0]  = np.exp(w[-1]-0.5*(v[-1]-v[0]))
    rate[0,0]   = - rate[1,0] - rate[-1,0]
    rate[-1,-1] = - rate[-2,-1] - rate[0,-1]

    # diagonal elements
    for i in range(1,n-1):
        rate[i,i] = - rate[i-1,i] - rate[i+1,i]
    return rate

def extend_profile(F,D,Drad,edges,mult):
    """Extend the profile by repeating it n=mult times"""
    F1 = np.array(F.tolist()*mult)
    D1 = np.array(D.tolist()*mult)
    Drad1 = np.array(Drad.tolist()*mult)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
    nK = mult*len(F)
    edges1 = np.arange(nK+1)*dz   # no longer symmetry about z=0
    return F1,D1,Drad1,edges1

def extend_profile_water(F,D,Drad,edges,extrawat):
    """Extend the profile by adding n=extrawat bins with properties of starting/ending bin"""
    # bin[0] and bin[-1] are usually the water layer, hence the name
    extraF    = [(F[-1]+F[0])*0.5 for i in range(extrawat)]
    extraD    = [D[-1] for i in range(extrawat)]
    extraDrad = [Drad[-1] for i in range(extrawat)]
    F1 = np.array(F.tolist()+extraF)
    D1 = np.array(D.tolist()+extraD)
    Drad1 = np.array(Drad.tolist()+extraDrad)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
    edges1 = np.arange(len(F)+1)*dz   # no longer symmetry about z=0
    #WOUTER: shouldn't it be F1??
    #dz = np.arange(len(F1)+1)*dz
    return F1,D1,Drad1,edges1

def extend_profile_water_LR(F,D,Drad,edges,extrawat):
    """Extend the profile by adding n=extrawat bins left and right of the membrane"""
    extraF = [(F[-1]+F[0])*0.5 for i in range(extrawat)]
    extraD = [D[-1] for i in range(extrawat)]
    extraDrad = [Drad[-1] for i in range(extrawat)]
    F1 = np.array(extraF+F.tolist()+extraF)
    D1 = np.array(extraD+D.tolist()+extraD)
    Drad1 = np.array(extraDrad+Drad.tolist()+extraDrad)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)
    edges1 = np.arange(len(F1)+1)*dz
    return F1,D1,Drad1,edges1

def just_get_me_my_F_and_D(fn, st=0, end=-1, dt=1.):
    """Read in and cut out from F and D profiles from filename fn"""
    F,D,edges = read_F_D_edges(fn)
    Drad = D # Not used, but just such that it has same dimensions.
    nbins = len(F)
    zpbc = edges[-1] - edges[0]
    dz = (edges[-1]-edges[0])/(len(edges)-1.) #angstrom
    v_short = F[st:end+1]
    d_short = D[st:end+1]
    drad_short = Drad[st:end+1]
    edges_short = edges[st:end+2] # has an extra element
    z_array = np.array([i*dz for i in range(len(d_short))])
    
    # Select the correct water bins
    # Yes, this introduces a discontinuity in the F and D profiles,
    # but it's a more realistic water profile.
    v_short[0] = F[0]
    v_short[-1] = F[-1]
    d_short[0] = D[0]
    d_short[-1] = D[-1]
    # idx -1 and 0 have the same values, don't worry.
    return v_short, d_short, drad_short, edges_short

def stack_me_and_add_water(v, d, drad, edges, m, nw_L, nw_R):
    """ We take an existing (v,d) profile (left-most and right-most bins 
    are assumed to be WATER and not part of the bilayer) and multiply the
    bilayer part m times. We then add nw_L and nw_R water bins to the left
    and right of the bilayer stack, respectively. Note that there will be
    nw_L water bins on the left, and not nw_L + 1. (Same for right side.)
    """
    # extend the bilayer m times
    vmulti, dmulti, dradmulti, edgesmulti =\
        extend_profile(v[1:-1], d[1:-1], drad[1:-1], edges[1:-1], m)
    # create left water bins
    vwl = [(v[-1]+v[0])*.5 for i in range(nw_L)]
    dwl = [(d[-1]+d[0])*.5 for i in range(nw_L)]
    dradwl = [(drad[-1] + drad[0])*.5 for i in range(nw_L)]
    # create right water bins
    vwr = [(v[-1]+v[0])*.5 for i in range(nw_R)]
    dwr = [(d[-1]+d[0])*.5 for i in range(nw_R)]
    dradwr = [(drad[-1] + drad[0])*.5 for i in range(nw_R)]
    # stack 'em
    vtot = np.array(vwl + vmulti.tolist() + vwr)
    dtot = np.array(dwl + dmulti.tolist() + dwr)
    dradtot = np.array(dradwl + dradmulti.tolist() + dradwr)

    return vtot, dtot, dradtot
    

### New functions: ###
def get_pbc_rate_matrix(F,D):
    n = len(F)
    rate = init_rate_matrix_pbc(n, F, D)
    return rate

def get_rate_matrix(F,D):
    n = len(F)
    rate = init_rate_matrix_pbc(n, F, D)
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
    # Sort from low to high, such that eigvec[:,-1] and 
    # eigvec[:,-2] are the lambda=0 eigvectors
    idx_sort = np.flip(eigval.argsort()[::-1])
    eigval = eigval[idx_sort]
    eigvec = eigvec[:,idx_sort]
    Diag = np.diag(eigval)
    Q = eigvec # R = Q.D.Q^-1, and thus exp(Rt) = Q.exp(Dt).Q^-1
    Qinv = np.linalg.inv(Q)
    max_diff = np.max(np.abs(np.matmul(np.matmul(Q,Diag),np.linalg.inv(Q))-R))
    print("||R - Q.D.Q^-1||_max = "+str(max_diff))
    return Diag,Q,Qinv

def propagate_with_diagonal(time, Diag, Q, Qinv, init=None):
    assert init is not None, "Please provide an initial distribution"
    # init = np.zeros(len(prop))
    # init[0]=1.
    Diag_exp = np.zeros_like(Diag)
    for i in range(len(Diag)):
        Diag_exp[i,i] = np.exp(Diag[i,i]*time)
    prop = np.matmul(np.matmul(Q,Diag_exp),Qinv)
    profile = np.dot(prop,init)
    return profile

def propagate_init_distrib_with_diagonal(time, Diag, Q, Qinv, init,
                                         init0=1., initN=0.):
    Diag_exp = np.zeros_like(Diag)
    for i in range(len(Diag)):
        Diag_exp[i,i] = np.exp(Diag[i,i]*time)
    prop = np.matmul(np.matmul(Q,Diag_exp),Qinv)
    init[0] = init0
    init[-1] = initN
    profile = np.dot(prop,init)
    return profile

def get_eigenrates(R):
    eigval, eigvec = np.linalg.eig(R)
    # Sort from low to high, such that eigvec[:,-1] and 
    # eigvec[:,-2] are the lambda=0 eigvectors
    idx_sort = np.flip(eigval.argsort()[::-1])
    eigval = eigval[idx_sort]
    eigvec = eigvec[:,idx_sort]
    return eigval, eigvec

def get_taus(D,F,dz=1.):
    mid = int(len(D)/2)
    left = int(len(D)/4)
    right = len(D)-left
    F_L = F[0]
    F_R = F[-1]
    R_L = np.sum(1./(D[1:mid]*np.exp(-(F[1:mid]-F_L))))*dz
    R_R = np.sum(1./(D[mid:-1]*np.exp(-(F[mid:-1]-F_R))))*dz
    R = np.sum(1./(D*np.exp(-(F-F_L))))*dz
    R_Lmin = np.sum(1./(D[1:left]*np.exp(-(F[1:left]-F_L))))*dz
    R_Rmin = np.sum(1./(D[right:-1]*np.exp(-(F[right:-1]-F_R))))*dz
    C = np.sum(np.exp(-(F[1:-1]-F_L)))*dz
    CL = np.sum(np.exp(-(F[1:-1]-F_L)))*dz
    CR = np.sum(np.exp(-(F[1:-1]-F_R)))*dz
    CmidR = np.sum(np.exp(-(F[left:right]-F_R)))*dz
    CmidL = np.sum(np.exp(-(F[left:right]-F_L)))*dz
    R_L_avg = np.sum([np.exp(-(F[i]-F_L))*np.sum(1./(D[1:i]*np.exp(-(F[1:i]-F_L))))*dz \
                      for i in range(1,mid)])/np.sum(np.exp(-(F[1:mid]-F_L)))
    R_R_avg = np.sum([np.exp(-(F[i]-F_R))*np.sum(1./(D[i:-1]*np.exp(-(F[i:-1]-F_R))))*dz \
                      for i in range(mid,len(F)-1)])/np.sum(np.exp(-(F[mid:-1]-F_R)))

    
    tau_mid = C*R_L*R_R/(R_L+R_R)
    tau_split = C*R_Lmin*R_Rmin/(R_Lmin+R_Rmin)
    tau_RC = C*R/4
    tau_new = (1/CL/R_L + 1/CR/R_R)**(-1)
    tau_newsplit = (1/CL/R_Lmin + 1/CR/R_Rmin)**(-1)
    tau_newest = (1/CL/R_L_avg + 1/CR/R_R_avg)**(-1)
    
    return [tau_mid,tau_split,tau_RC,tau_new,tau_newsplit,tau_newest]
    
def get_new_states(B,P,D,t):
    Dinv = np.linalg.inv(D)
    I = np.identity(len(P))
    Pinv = np.linalg.inv(P)
    D_exp = np.zeros_like(D)
    for i in range(len(D_exp)):
        D_exp[i,i] = np.exp(D[i,i]*t)
    y_from_sources = -1*np.matmul(np.matmul(np.matmul(np.matmul(P,
                                                                Dinv),
                                                                I-D_exp),
                                                                Pinv),
                                                                B)
    return y_from_sources
    
def get_new_states_with_x0(B,P,D,t,x0=None):
    Dinv = np.linalg.inv(D)
    I = np.identity(len(P))
    Pinv = np.linalg.inv(P)
    D_exp = np.zeros_like(D)
    for i in range(len(D_exp)):
        D_exp[i,i] = np.exp(D[i,i]*t)
    y_from_sources = -1*np.matmul(np.matmul(np.matmul(np.matmul(P,
                                                                Dinv),
                                                                I-D_exp),
                                                                Pinv),
                                                                B)
    y_from_states = np.matmul(np.matmul(np.matmul(P,
                                                  D_exp),
                                                  Pinv),
                                                  x0)
    return y_from_sources + y_from_states
