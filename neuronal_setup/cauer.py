import numpy as np

def get_An(n,RC=1):
    if n == 1:
        return np.array([-2])
    if n == 2:
        return np.array([[-3/2,1/2],[1/2,-3/2]])
    
    mat = np.zeros([n,n],float)
    for d in range(n):
        mat[d,d] = -1
    mat[0,0] = -1.5
    mat[n-1,n-1] = -1.5
    for j in range(n):
        if j == 0:
            mat[j,j+1] = 0.5
        elif j == n-1:
            mat[j,j-1] = 0.5
        else:
            mat[j,j+1] = 0.5
            mat[j,j-1] = 0.5
    if RC != 1:
        mat = mat*(RC**(-1))
    return mat

def get_eigen_from_matrix(n,RC=1):
    """
        Returns the eigenvalues and corresponding eigenvectors of A, 
        sorted for increasing eigenvalues (so lowest lambda first)
    """
    A = get_An(n)
    evals, evecs = np.linalg.eig(A)
    sorted_ids = np.argsort(evals)
    if RC != 1:
        return evals[sorted_ids]*(RC**(-1)), evecs[:,sorted_ids]
    return evals[sorted_ids], evecs[:,sorted_ids]
    

def get_eigen_from_model(n, RC=1):
    evals = np.array([(-1. + np.cos(i*np.pi/n)) for i in range(1,n+1)])
    evecs = np.zeros([n,n],float)
    for i in range(n):
        for j in range(n-1):
            evecs[i,j] = np.sin((j+1)*(2*(i+1)-1)*np.pi/2/n)
    
    for i in range(n):
        evecs[i,-1] = (-1.)**i # because it is rho**(i-1), i = 1 .. n
    
    for i in range(n):
        evecs[:,i] = evecs[:,i]/np.linalg.norm(evecs[:,i])
        
    sorted_ids = np.argsort(evals)
    if RC != 1:
        return evals[sorted_ids]*(RC**(-1)), evecs[:,sorted_ids]
    else:
        return evals[sorted_ids], evecs[:,sorted_ids]

def get_sources(Vl = 1., Vr = 0):
    return np.array([Vl,Vr])

def get_Bn(n, RC=1):
    Bl = RC**(-1)
    Br = RC**(-1)
    B = np.zeros([n,2],float)
    B[0,0] = Bl
    B[n-1,1] = Br
    return B

def get_states_at_t(B,P,D,V,t):
    n = len(D)
    Dinv = np.linalg.inv(D)
    I = np.identity(n)
    Pinv = np.linalg.inv(P)
    D_exp = np.zeros_like(D)
    for i in range(len(D_exp)):
        D_exp[i,i] = np.exp(D[i,i]*t)
    #return np.matmul(P,np.matmul(Dinv,np.matmul(I-D_exp,np.matmul(Pinv,B))))
    return -1.*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(P,Dinv),I-D_exp),Pinv),B),V)
    
def get_states_at_t_with_x0(B,P,D,V,t,x0):
    n = len(D)
    Dinv = np.linalg.inv(D)
    I = np.identity(n)
    Pinv = np.linalg.inv(P)
    D_exp = np.zeros_like(D)
    for i in range(len(D_exp)):
        D_exp[i,i] = np.exp(D[i,i]*t)
    #return np.matmul(P,np.matmul(Dinv,np.matmul(I-D_exp,np.matmul(Pinv,B))))
    return -1.*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(P,Dinv),I-D_exp),Pinv),B),V) + np.matmul(np.matmul(np.matmul(P,D_exp),Pinv),x0)

def get_steady_state(B,P,D,V):
    Dinv = np.linalg.inv(D)
    Pinv = np.linalg.inv(P)
    return -1.*np.matmul(np.matmul(np.matmul(np.matmul(P,Dinv),Pinv),B),V)
