import numpy as np
import scipy.linalg as scilin



def SpectrogramSeparation(Sz,K,alpha,beta,gamma,Niter):
    
    N,M  = np.shape(Sz)
    
    c = np.zeros(N-1)
    c[0]  = -1
    r = np.zeros(N)
    r[0:2] = [-1, 1]
    B = scilin.toeplitz(c,r)
    
    Mat = np.eye(N) + alpha * (B.T @ B)
    
    Sy = np.zeros((N,M))
    L = np.zeros(K)
    for k  in range(K):  
        Sx = np.linalg.solve(Mat, Sz-Sy)      
        Sy = solveSy(Sz-Sx,gamma,Niter,Sy,beta)
        
        L[k] = costFunction(Sz,Sx,Sy,alpha,beta,B)
    
    return  Sx,Sy,L


def solveSy(S,gamma,Niter,x,beta):
    
    t = 1
    z = x
    for  ind_iter in range(Niter):
        xnew = proxL1(z -  2*gamma*(z-S), gamma*beta )
        tnew = (1+np.sqrt(4*t**2+1))/2
        eta = (t-1)/tnew
        z = xnew + eta * (xnew-x)
        
        x = xnew
        t = tnew
        
    return x

def proxL1(u,xi):
    v = np.abs(u)-xi 
    tmp =  np.where(v>0, v, 0)
    v  = np.sign(u) * tmp
    return v

def costFunction(Sz,Sx,Sy,alpha,beta,B):
    term1 = np.linalg.norm(Sz-Sx-Sy)
    term2 = alpha * np.linalg.norm( B @ Sx )
    term3 = beta * np.linalg.norm(Sy.flatten(),1)
    
    return term1 + term2  + term3
    