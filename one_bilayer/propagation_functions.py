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

def propagate_init_distrib_with_diagonal(time,Diag,Q,Qinv,init,init0=1.,initN=0.):
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
    # Sort from low to high, such that eigvec[:,-1] and eigvec[:,-2] are the lambda=0 eigvectors
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
    R_L_avg = np.sum([np.exp(-(F[i]-F_L))*np.sum(1./(D[1:i]*np.exp(-(F[1:i]-F_L))))*dz for i in range(1,mid)])/np.sum(np.exp(-(F[1:mid]-F_L)))
    R_R_avg = np.sum([np.exp(-(F[i]-F_R))*np.sum(1./(D[i:-1]*np.exp(-(F[i:-1]-F_R))))*dz for i in range(mid,len(F)-1)])/np.sum(np.exp(-(F[mid:-1]-F_R)))

    
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
    y_from_sources = -1*np.matmul(np.matmul(np.matmul(np.matmul(P,Dinv),I-D_exp),Pinv),B)
    return y_from_sources
    
def get_new_states_with_x0(B,P,D,t,x0=None):
    Dinv = np.linalg.inv(D)
    I = np.identity(len(P))
    Pinv = np.linalg.inv(P)
    D_exp = np.zeros_like(D)
    for i in range(len(D_exp)):
        D_exp[i,i] = np.exp(D[i,i]*t)
    y_from_sources = -1*np.matmul(np.matmul(np.matmul(np.matmul(P,Dinv),I-D_exp),Pinv),B)
    y_from_states = np.matmul(np.matmul(np.matmul(P,D_exp),Pinv),x0)
    return y_from_sources + y_from_sta