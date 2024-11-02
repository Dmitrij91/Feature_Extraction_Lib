import cython
import numpy as np
cimport numpy as cnp
import time
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport exp, pi,pow,sqrt,abs
cimport scipy.linalg.cython_lapack as lapack
from scipy.linalg.cython_lapack cimport dpotrf as cholesky_c
from scipy.linalg.cython_lapack cimport dtrtri as Upper_inv
from scipy.linalg.cython_lapack cimport dhseqr as schur_c
from scipy.linalg.cython_lapack cimport dgehrd as hessenberg_c_1 
from scipy.linalg.cython_lapack cimport dorghr as Q_transform # Get the Q Transform from vector Q
from scipy.linalg.cython_lapack cimport zpotrf,dsyev

'Import C-Functions'

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double fmin(double x, double y) nogil
    double fmax(double x, double y) nogil
    double fabs(double x) nogil


cdef extern from 'complex.h' nogil:
    double creal(complex)
    
ctypedef fused TYPE:
    double
    complex

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
    
cdef inline double logdet(TYPE* X, int s, int* info) nogil:
    # Computes log(det(A)) for a matrix X
    cdef double out = 0.0
    cdef int i
    if TYPE is double:
        cholesky_c('L', &s, X, &s, info)
        for i in range(s):
            out += 2.0 * log(X[i*(s+1)])
    elif TYPE is complex:
        zpotrf('L', &s, X, &s, info)
        for i in range(s):
            out += 2.0 * log(creal(X[i*(s+1)]))
    return out

""" References \
 [1] D.A. Bini and B. Iannazzo, "Computing the Karcher mean of symmetric \
 positive definite matrices", Linear Algebra Appl., 438-4 (2013), """

@cython.boundscheck(True)
@cython.wraparound(True)
@cython.cdivision(True)


def Bini_Riemann(cnp.ndarray[double,ndim = 3,negative_indices = True] Matrix_Array, int Max_Iter = 120,\
                                                              double tol = 1e-4,bint Aut = True):
    'Initiazation'
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    cdef int N = Matrix_Array.shape[0]
    cdef int d = Matrix_Array.shape[1]
    cdef cnp.ndarray[double,ndim = 3,negative_indices = False] List_Cholesky = np.zeros((N,d,d)) 
    cdef int nuold = 100000
    
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] Mean = np.zeros((d,d))
    cdef int k,info,j,i,l,it
    cdef double theta, beta, ch, gamma ,dh
    
    'Declare Variables for the main iteration'
    
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] R0       = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] S        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] R0_Store = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] V        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] U        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] Q        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] T        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] iR0      = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] Z        = np.zeros((d,d))
    cdef cnp.ndarray[double, ndim = 3,negative_indices = False] U_d      = np.zeros((N,d,d))
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] V_d      = np.zeros((N,d))
    
    'For symmetric matrices only storage of lower or upper triangular part needed'
    
    for k in range(N): 
        for i in range(d):
            for j in range(i,d):
                List_Cholesky[k,i,j] = Matrix_Array[k,i,j]
                
        'Compute Cholesky Decomposition of List Cholesky'
        
        # Give Pointer to the fist variable of List_Cholesky not the whole matrix
        
        cholesky_c('L',&d,&List_Cholesky[k,0,0],&d,&info) 
        assert info == 0
                
    
    'Initialization with Arithmetic Mean'
    
    for k in range(N):
        for i in range(d):
            for j in range(d):
                Mean[i,j] = Mean[i,j]+Matrix_Array[k,i,j]/N
    Mean = np.mean(Matrix_Array,axis = 0 )
    
    
    ' Main Iteration '
    starttime = time.time()
    for it in range(Max_Iter):
        
        ' Cholseky Dec. of current Mean Iterate '
        
        R0   = Return_Cholesky(Mean) 
        
        'Inverse of Cholesky'
        
        iR0  = Inv_Upper(R0)  
    
        
        'Initilize List for Cholesky '
        
        
        for k in range(N):
            Z        = List_Cholesky[k]@iR0
            V,U      = Schur_Cython(Z.T@Z) # Pass Z.T T as U_d[k] to ietration for better Performance
            U_d[k]   = U
            V_d[k]   = np.diag(V)
        
        ' Routine for adaptive step size selection as preconditing with largest and smalles eigenvalue '
        
        if Aut == True:
            beta  = 0
            gamma = 0
            for j in range(N):
                ch = np.max(V_d[j])/np.min(V_d[j])
                if np.abs(ch-1) < 0.5:
                    dh = np.log1p(ch-1)/(ch-1)
                else:
                    dh = np.log(ch)/(ch-1)
                beta  = beta +  dh
                gamma = gamma+ ch*dh
            theta = 2/(gamma + beta)
        
        T = U_d[0]@np.diag(np.log(V_d[0]))@U_d[0].T
        S =  (T+(T.T))/2
        
        for k in range(1,N):
            T = U_d[k]@np.diag(np.log(V_d[k]))@U_d[k].T
            S = S + (T+(T.T))/2
        Vs,Us = Schur_Cython(S)
        Z     = np.diag(np.exp(np.diag(Vs*theta/2)))@(Us.T)@R0
        Mean  = Z.T@Z
        
        'Compute norm of S'
    
        nu = np.sqrt(np.mean(np.abs(Mean) ** 2))
        if nu < tol or nu > nuold:
 
            t_end = time.time()-starttime
            print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
            break
        nuold = nu
            
    return Mean,,t_end


def Riemann_Update_Cluster(w, Data, Clusters):
    
    """Returns Riemannian Means """
    
    # sanity checks:
    assert Data.shape[1] == Clusters.shape[2] == Clusters.shape[1] == Data.shape[2]
    assert w.shape[0] == Data.shape[0] and w.shape[1] == Clusters.shape[0]
    
    ' Allocate Memory for Riemannian Clusters '
    
    K = w.shape[1]
    d = Data.shape[1]
    
    out = np.zeros((K,d,d))
    
    
    for j in range(K):
        print(w)
        Mask = Data[w[:,j] == 1]
        print(Mask.shape)
        if Mask.shape[0] !=0:
            out[j,:,:] = Bini_Riemann(Mask)
              

    return np.asarray(out)





def distance_Riemmann(cnp.ndarray[double,ndim = 3,negative_indices = False] data,\
                      cnp.ndarray[double,ndim = 3,negative_indices = False] prototype, 
                      double[:,:] out = None):
    
    """Computes the matrix of Riemmannian distances between data and prototypes."""
    
    # sanity checks:
    assert data.shape[1] == data.shape[2]
    assert prototype.shape[1] == prototype.shape[2]
    assert data.shape[1] == prototype.shape[1]
    
    ' Descriptor shape '
    
    cdef int d = data.shape[1]
    
    
    ' Datasize '
    
    cdef int m = data.shape[0]
    
    ' Number of Prototypes '
    
    cdef int K = prototype.shape[0]
    
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False] help_i = np.zeros((d,d)) 
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False] help_j = np.zeros((d,d)) 
    
    'Initilize Memory for Output of size m times K '
    
    if out is None:
        out = np.empty((m,K))
        
    
    #cdef int tid, num_threads = omp_get_max_threads()
    
    #print('Start Computation with maximal number of  -------{:*^10}------- Threads'.format(num_threads))
    
    'TODO --> Rewrite Riemmann distance with nogil avoiding Python object and execute the below loop in parallel'
    
    ' Allocate Memory for Covariance data and Prototype Array within parallel loop '
    
    #cdef double* Nodes_Cov = <double*> malloc(sizeof(double) * d*d * num_threads)
    #cdef double* Prot_Cov = <double*> malloc(sizeof(double) * d*d*K * num_threads)
   # cdef double* Pointer_Data
   # cdef double* Pointer_Prot
    
    cdef int i, j, info
    
        
    'Iterate over Nodes in parallel using omp_get_thread_num() for current iterator index '
                 
    for i in range(m):
            
        ' Copy Covariance matrices from data into shared memory by filling the rowes of size d times d '
            
        ' Iterate over Prototypes '
            
        for j in range(K):
            out[i,j] = Riemmannian_Distance_Cython(data[i,:,:],prototype[j,:,:])
    

    return np.asarray(out)


from scipy.linalg.cython_lapack cimport dsteqr as Eigv_Trid # tridiagonal real
from scipy.linalg.cython_lapack cimport dsytrd as Reduce_Sym 
from scipy.linalg.cython_lapack cimport dsygst as Standard_Form


' Computes the Riemannian Distance through generalized eigenvalue dec '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Riemmannian_Distance_Cython(cnp.ndarray[double, ndim=2, negative_indices=False] A, \
                         cnp.ndarray[double, ndim=2, negative_indices=False] B):
  
    # sanity checks:
    assert A.shape[0] == A.shape[1]
    assert B.shape[0] == B.shape[1]
    assert A.shape[0] == B.shape[0]
    
    
    cdef int d = A.shape[1]
    
    
    cdef int info,j,i
    cdef double Distance = 0
    
    cdef int lwork       = d
    
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False]  Work            = np.empty((lwork)) 
    #cdef double[:,:] Chol_B = cvarray(shape=(d,d),itemsize=sizeof(double),format='d')
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False]  Reduced_C       = np.zeros((d,d))
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False]  Chol_B          = B.copy()
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False]  Diag_C          = np.zeros((d)) 
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False]  Off_Diag_C      = np.zeros((d-1))
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False]  Tau_C           = np.zeros((d-1))
    
    cdef int TYPE = 1 # for Ax = lambda Bx
    
    'Compute Cholesky of B '
    
    cholesky_c('L',&d,&Chol_B[0,0],&d,&info)
    
    for i in range(d):
        for j in range(i,d):
            Reduced_C[i,j] = A[i,j]
    
    ' Transform Ax = lambda Bx to standard form Cy = lambda y '
    
    Standard_Form(&TYPE,'L',&d,&Reduced_C[0,0],&d,&Chol_B[0,0],&d,&info)
    assert info == 0
    
    ' Tridiagonal Reduction of C '
    
    Reduce_Sym('L',&d,&Reduced_C[0,0],&d,&Diag_C[0],&Off_Diag_C[0],&Tau_C[0],&Work[0],&lwork,&info)
    assert info == 0
    
    ' Compute Generalized Eigenvalues '
    
    Eigv_Trid('N',&d,&Diag_C[0],&Off_Diag_C[0],&A[0,0],&d,&Work[0],&info)
    assert info == 0
    
    for i in range(d):
        Distance = Distance  + (log(Diag_C[i]))*(log(Diag_C[i]))
        
    return sqrt(Distance)
    

' Inverse of upper triangular matrix '

@cython.boundscheck(False)
@cython.wraparound(False)


def Inv_Upper(cnp.ndarray[double,ndim = 2,negative_indices = False] A):
    assert A.shape[0] == A.shape[1]
    
    cdef int d = A.shape[1]
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False] Inv_Chol = np.zeros((d,d)) 
    cdef int info,j,i
    
    for i in range(d):
        for j in range(i,d):
            Inv_Chol[i,j] = A[i,j]
    
    Upper_inv('L','N',&d,&Inv_Chol[0,0],&d,&info)
    
    return Inv_Chol


' Function for Get Upper Cholesky Dec '

@cython.boundscheck(False)
@cython.wraparound(False)

def Return_Cholesky(cnp.ndarray[double,ndim = 2,negative_indices = False] A):
    assert A.shape[0] == A.shape[1]
    
    cdef int d = A.shape[1]
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False] work = np.zeros((d,d)) 
    cdef int info,j,i
    
    for i in prange(d,nogil = True):
        for j in range(i,d):
            work[i,j] = A[i,j]
    
    cholesky_c('L',&d,&work[0,0],&d,&info)
    
    return work
    
' Schur Decomposition for a Upper triangular Matrix quadratic '
' The Computation is performed on input matrix A !!!! if separate matrix needed initilize an perform compuation on copy'
' Repace work with A and vice versa '

@cython.boundscheck(False)
@cython.wraparound(False)

def Schur_Cython(cnp.ndarray[double,ndim = 2,negative_indices = False] A):
    
    'Initiazation'
    
    assert A.shape[0] == A.shape[1]
    
    cdef int d = A.shape[1]
    cdef cnp.ndarray[double,ndim = 2,negative_indices = False] work = np.zeros((d,d))  
    cdef cnp.ndarray[double, ndim = 1,negative_indices = False] WR = np.zeros((d))
    cdef cnp.ndarray[double, ndim = 1,negative_indices = False] WI = np.zeros((d))
    
    
    'Here the The scalars of Householder reflection are stored'
    
    cdef cnp.ndarray[double, ndim = 1,negative_indices = False] Tau = np.zeros((d-1))
    
    
    cdef int info,j,i
    
    "ILO = lowest index where the original matrix had been Hessenbergform \ "
    "IHI = highest index where the original matrix had been Hessenbergform"
    
    cdef int ILO       = 1
    cdef int IHI       = d
    cdef int lwork     = d
    cdef int LDA       = d
    
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False] Work = np.empty((lwork))
    
    for i in prange(d,nogil = True):
        for j in range(d):
            work[i,j] = A[i,j]
    
    'Compute the Hessenberg form of A '
    
    hessenberg_c_1(&d,&ILO,&IHI,&work[0,0],&LDA,&Tau[0],&Work[0],&lwork,&info)
    
    'Get the Matrix Q'
    
    # Initialization
    
    cdef cnp.ndarray[double, ndim = 2,negative_indices = False] Q = np.zeros((d,d))
    
    for i in prange(d,nogil = True):
        Q[0,i] = work[0,i] 
    
    for i in prange(1,d,nogil = True):
        for j in range(i-1,d):
            Q[i,j] = work[i,j]
    
    Q_transform(&d,&ILO,&IHI,&Q[0,0],&LDA,&Tau[0],&Work[0],&lwork,&info)
    
    'Compute the QR factorization of A applied on similar matrix H computed above'
    
    schur_c('S','V',&d,&ILO,&IHI,&work[0,0],&d,&WR[0],&WI[0],&Q[0,0],&d,&Work[0],&lwork,&info)
    
    return work.T,Q.T

@cython.boundscheck(False)
@cython.wraparound(False)

def expmh(double[:,:] M, double[:,:] out):
    ### Matrix exponential of a symmetric matrix ###
    cdef int n = M.shape[0], lwork = 3*M.shape[0]-1, info, i, j, k
    cdef double* V = <double*> malloc(sizeof(double) * n * n)
    cdef double* w = <double*> malloc(sizeof(double) * n)
    cdef double* work = <double*> malloc(sizeof(double) * lwork)
    # Copy upper triangular part of M:
    for i in range(n):
        for j in range(i,n):
            V[n*i+j] = M[i,j]
    # Eigenvalue decomposition:
    dsyev('V', 'L', &n, V, &n, w, work, &lwork, &info)
    assert info == 0
    # Exp(M) using eigenvalue decomposition:
    for i in range(n):
        for j in range(i,n):
            out[i,j] = 0.0
            for k in range(n):
                out[i,j] += exp(w[k]) * V[n*k+i] * V[n*k+j]
    # Symmetrize result:
    for j in range(n):
        for i in range(j+1,n):
            out[i,j] = out[j,i]
    # Free allocated memory:
    free(V)
    free(w)
    free(work)
    return np.asarray(out)

@cython.boundscheck(False)
@cython.wraparound(False)
def logmh(cnp.ndarray[double, ndim=2, negative_indices=False] M):
    assert M.shape[0] == M.shape[1]
    cdef int n = M.shape[0], lwork = 3*M.shape[0]-1, info, i, j, k
    cdef cnp.ndarray[double, ndim=2, negative_indices=False] V = np.empty((n,n))
    cdef cnp.ndarray[double, ndim=2, negative_indices=False] log_M = np.empty((n,n))
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] w = np.empty((n,))
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] work = np.empty((lwork,))
    # Copy upper triangular part of M:
    for i in range(n):
        for j in range(i,n):
            V[i,j] = M[i,j]
    # Eigenvalue decomposition:
    lapack.dsyev('V', 'L', &n, &V[0,0], &n, &w[0], &work[0], &lwork, &info)
    assert info == 0
    # Log(M) using eigenvalue decomposition:
    for i in range(n):
        for j in range(i,n):
            log_M[i,j] = 0.0
            for k in range(n):
                log_M[i,j] += log(w[k]) * V[k,i] * V[k,j]
    # Symmetrize result:
    for j in range(n):
        for i in range(j+1,n):
            log_M[i,j] = log_M[j,i]
    return log_M



@cython.boundscheck(False)
@cython.wraparound(False)
def sqrtmh(cnp.ndarray[double, ndim=2, negative_indices=False] M):
    assert M.shape[0] == M.shape[1]
    cdef int n = M.shape[0], lwork = 3*M.shape[0]-1, info, i, j, k
    cdef cnp.ndarray[double, ndim=2, negative_indices=False] V = np.empty((n,n))
    cdef cnp.ndarray[double, ndim=2, negative_indices=False] sqrt_M = np.empty((n,n))
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] w = np.empty((n,))
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] work = np.empty((lwork,))
    # Copy upper triangular part of M:
    for i in range(n):
        for j in range(i,n):
            V[i,j] = M[i,j]
    # Eigenvalue decomposition:
    lapack.dsyev('V', 'L', &n, &V[0,0], &n, &w[0], &work[0], &lwork, &info)
    assert info == 0
    # Sqrt(M) using eigenvalue decomposition:
    for i in range(n):
        for j in range(i,n):
            sqrt_M[i,j] = 0.0
            for k in range(n):
                sqrt_M[i,j] += sqrt(w[k]) * V[k,i] * V[k,j]
    # Symmetrize result:
    for j in range(n):
        for i in range(j+1,n):
            sqrt_M[i,j] = sqrt_M[j,i]
    return sqrt_M



@cython.boundscheck(False)
@cython.wraparound(False)
def invh(cnp.ndarray[double, ndim=2, negative_indices=False] M):
    assert M.shape[0] == M.shape[1]
    cdef int n = M.shape[0], info, i, j
    cdef cnp.ndarray[double, ndim=2, negative_indices=False] inv_M = np.empty((n,n))
    # Copy upper triangular part of M:
    for i in range(n):
        for j in range(i,n):
            inv_M[i,j] = M[i,j]
    # Cholesky decomposition:
    lapack.dpotrf('L', &n, &inv_M[0,0], &n, &info)
    assert info == 0
    # Matrix inverse using Cholesky decomposition:
    lapack.dpotri('L', &n, &inv_M[0,0], &n, &info)
    assert info == 0
    # Symmetrize result:
    for j in range(n):
        for i in range(j+1,n):
            inv_M[i,j] = inv_M[j,i]
    return inv_M