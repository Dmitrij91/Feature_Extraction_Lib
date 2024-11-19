import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from openmp cimport omp_get_max_threads, omp_get_thread_num
from scipy.linalg.cython_blas cimport dtrsm, dgemv, ddot
from scipy.linalg.cython_lapack cimport dgelqf, dorglq
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dpotri, zpotrf, zpotri
from libc.math cimport log, cos, sin, pi
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy



cdef extern from 'complex.h' nogil:
    double creal(complex)


    
ctypedef fused TYPE:
    double
    complex



cdef inline double logdet(TYPE* X, int s, int* info) nogil:
    # Computes log(det(A)) for a matrix X
    cdef double out = 0.0
    cdef int i
    if TYPE is double:
        dpotrf('L', &s, X, &s, info)
        for i in range(s):
            out += 2.0 * log(X[i*(s+1)])
    elif TYPE is complex:
        zpotrf('L', &s, X, &s, info)
        for i in range(s):
            out += 2.0 * log(creal(X[i*(s+1)]))
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
def distance_stein(TYPE[:,:,:] data, TYPE[:,:,:] prototype, 
                   double[:,:] out = None):
    """Computes the matrix of Stein distances between data and prototypes."""
    
    # sanity checks:
    assert data.shape[1] == data.shape[2]
    assert prototype.shape[1] == prototype.shape[2]
    assert data.shape[1] == prototype.shape[1]
    
    cdef int d = data.shape[1], m = data.shape[0], n = prototype.shape[0]
    if out is None:
        out = np.empty((m,n))
    cdef double* logdet_d = <double*> malloc(sizeof(double) * m)
    cdef double* logdet_p = <double*> malloc(sizeof(double) * n)
    
    cdef int tid, num_threads = omp_get_max_threads()
    cdef TYPE* work = <TYPE*> malloc(sizeof(TYPE) * d*d * num_threads)
    cdef TYPE* Z
    
    cdef int i, j, k, l, info
    with nogil:
        # Compute logdet_d (= logdet of data):
        for i in prange(m, schedule='static'):
            tid = omp_get_thread_num()
            Z = &work[d*d*tid]
            # copy upper triangular part into working space:
            for k in range(d):
                for l in range(k,d):
                    Z[d*k+l] = data[i,k,l]
            logdet_d[i] = logdet(Z, d, &info)
        # Compute logdet_p (= logdet of prototype):
        for j in prange(n, schedule='static'):
            tid = omp_get_thread_num()
            Z = &work[d*d*tid]
            # Copy upper triangular part into work space:
            for k in range(d):
                for l in range(k,d):
                    Z[d*k+l] = prototype[j,k,l]
            logdet_p[j] = logdet(Z, d, &info)
        # Finally compute distance matrix:
        for i in prange(m, schedule='static'):
            tid = omp_get_thread_num()
            Z = &work[d*d*tid]
            for j in range(n):
                # Copy upper triangular part of (data[i] + prototype[j])/2 
                # into working space:
                for k in range(d):
                    for l in range(k,d):
                        Z[d*k+l] = 0.5 * (data[i,k,l] + prototype[j,k,l])
                out[i,j] = logdet(Z, d, &info) \
                            - 0.5 * logdet_d[i] - 0.5 * logdet_p[j]
    
    free(work)
    free(logdet_d)
    free(logdet_p)
    return np.asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
def riccatiG(double[:,:] w, TYPE[:,:,:] F, TYPE[:,:,:] G, 
             TYPE[:,:,:] out = None):
    """Computes: out[j,:,:] = 2 * sum_i w[i,j] * inv( F[i,:,:] + G[j,:,:] )"""
    
    # sanity checks:
    assert F.shape[1] == F.shape[2] == G.shape[1] == G.shape[2]
    assert w.shape[0] == F.shape[0] and w.shape[1] == G.shape[0]
    
    cdef int m = w.shape[0], n = w.shape[1], d = F.shape[1]
    
    cdef int tid, num_threads = omp_get_max_threads()
    cdef TYPE* work = <TYPE*> malloc(sizeof(TYPE) * d*d * num_threads)
    cdef TYPE* Z
    
    cdef int i, j, k, l, info
    if out is None:
        if TYPE is double:
            out = np.empty((n,d,d))
        elif TYPE is complex:
            out = np.empty((n,d,d), dtype=np.complex128)
    for j in prange(n, nogil=True):
        tid = omp_get_thread_num()
        Z = &work[d*d*tid]
        # initialize with 0s:
        for k in range(d):
            for l in range(k,d):
                out[j,k,l] = 0.0
        for i in range(m):
            # Copy upper triangular part into working space:
            for k in range(d):
                for l in range(k,d):
                    Z[d*k + l] = F[i,k,l] + G[j,k,l]
            if TYPE is double:
                # Cholesky decomposition:
                dpotrf('L', &d, Z, &d, &info)
                # Matrix inverse using Cholesky decomposition:
                dpotri('L', &d, Z, &d, &info)
            elif TYPE is complex:
                # Cholesky decomposition:
                zpotrf('L', &d, Z, &d, &info)
                # Matrix inverse using Cholesky decomposition:
                zpotri('L', &d, Z, &d, &info)
            # add inverse to result:
            for k in range(d):
                for l in range(k,d):
                    out[j,k,l] += 2.0 * w[i,j] * Z[d*k + l]
        # Symmetrize result:
        for l in range(d):
            for k in range(l+1,d):
                out[j,k,l] = out[j,l,k]
    free(work)
    return np.asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double logdet_3x3(int i, TYPE[:,:,:] F) nogil:
    # Computes log(det(F[i])) for the 3x3 matrix F[i]
    cdef TYPE tmp = F[i,0,0]*F[i,1,1]*F[i,2,2] + F[i,0,1]*F[i,1,2]*F[i,2,0] \
                    + F[i,0,2]*F[i,1,0]*F[i,2,1] - F[i,2,0]*F[i,1,1]*F[i,0,2] \
                    - F[i,1,0]*F[i,0,1]*F[i,2,2] - F[i,0,0]*F[i,2,1]*F[i,1,2]
    if TYPE is double:
        return log(tmp)
    elif TYPE is complex:
        return log(creal(tmp))
    


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double logdetmean_3x3(int i, int j, 
                                  TYPE[:,:,:] F, TYPE[:,:,:] G) nogil:
    # Computes log(det(0.5*(F[i]+G[j]))) for 3x3 matrices F[i] and G[j]
    cdef TYPE tmp = \
        (F[i,0,0]+G[j,0,0]) * (F[i,1,1]+G[j,1,1]) * (F[i,2,2]+G[j,2,2]) \
        + (F[i,0,1]+G[j,0,1]) * (F[i,1,2]+G[j,1,2]) * (F[i,2,0]+G[j,2,0]) \
        + (F[i,0,2]+G[j,0,2]) * (F[i,1,0]+G[j,1,0]) * (F[i,2,1]+G[j,2,1]) \
        - (F[i,2,0]+G[j,2,0]) * (F[i,1,1]+G[j,1,1]) * (F[i,0,2]+G[j,0,2]) \
        - (F[i,1,0]+G[j,1,0]) * (F[i,0,1]+G[j,0,1]) * (F[i,2,2]+G[j,2,2]) \
        - (F[i,0,0]+G[j,0,0]) * (F[i,2,1]+G[j,2,1]) * (F[i,1,2]+G[j,1,2])
    if TYPE is double:
        return log(0.125 * tmp)
    elif TYPE is complex:
        return log(0.125 * creal(tmp))



@cython.boundscheck(False)
@cython.wraparound(False)
def distance_stein_3x3(TYPE[:,:,:] F, TYPE[:,:,:] G, double[:,:] out = None):
    """Computes the matrix of Stein distances between 3x3 matrices."""
    
    # sanity checks:
    assert F.shape[1] == F.shape[2] == G.shape[1] == G.shape[2] == 3
    
    cdef int m = F.shape[0], n = G.shape[0]
    cdef double* ldF = <double*> malloc(sizeof(double) * m)
    cdef double* ldG = <double*> malloc(sizeof(double) * n)
    cdef int i, j
    if out is None:
        out = np.empty((m,n))
    
    for i in prange(m, nogil=True):
        ldF[i] = logdet_3x3(i, F)
    
    for j in prange(n, nogil=True):
        ldG[j] = logdet_3x3(j, G)
    
    for i in prange(m, nogil=True):
        for j in range(n):
            out[i,j] = logdetmean_3x3(i, j, F, G) - 0.5 * ldF[i] - 0.5 * ldG[j]
    free(ldF)
    free(ldG)
    return np.asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def riccatiG_3x3(double[:,:] w, TYPE[:,:,:] F, TYPE[:,:,:] G, 
                 TYPE[:,:,:] out = None):
    """Computes: out[j,:,:] = 2 * sum_i w[i,j] * inv( F[i,:,:] + G[j,:,:] )"""
    
    # sanity checks:
    assert F.shape[1] == F.shape[2] == G.shape[1] == G.shape[2] == 3
    assert w.shape[0] == F.shape[0]
    assert w.shape[1] == G.shape[0]
    if out is not None:
        assert out.shape[1] == out.shape[2] == 3
        assert out.shape[0] == G.shape[0]
    
    cdef int m = F.shape[0], n = G.shape[0]
    cdef int i, j
    cdef TYPE det
    if out is None:
        if TYPE is double:
            out = np.empty((n,3,3))
        elif TYPE is complex:
            out = np.empty((n,3,3), dtype=np.complex128)
    
    for j in prange(n, nogil=True):
        out[j,0,0] = out[j,0,1] = out[j,0,2] = out[j,1,0] = out[j,1,1] \
            = out[j,1,2] = out[j,2,0] = out[j,2,1] = out[j,2,2] = 0.0
        for i in range(m):
            det = (F[i,0,0]+G[j,0,0])*(F[i,1,1]+G[j,1,1])*(F[i,2,2]+G[j,2,2]) \
                + (F[i,0,1]+G[j,0,1])*(F[i,1,2]+G[j,1,2])*(F[i,2,0]+G[j,2,0]) \
                + (F[i,0,2]+G[j,0,2])*(F[i,1,0]+G[j,1,0])*(F[i,2,1]+G[j,2,1]) \
                - (F[i,2,0]+G[j,2,0])*(F[i,1,1]+G[j,1,1])*(F[i,0,2]+G[j,0,2]) \
                - (F[i,1,0]+G[j,1,0])*(F[i,0,1]+G[j,0,1])*(F[i,2,2]+G[j,2,2]) \
                - (F[i,0,0]+G[j,0,0])*(F[i,2,1]+G[j,2,1])*(F[i,1,2]+G[j,1,2])
            out[j,0,0] += \
                2.0 * w[i,j] * ( (F[i,1,1]+G[j,1,1]) * (F[i,2,2]+G[j,2,2]) 
                    - (F[i,1,2]+G[j,1,2]) * (F[i,2,1]+G[j,2,1]) ) / det
            out[j,0,1] += \
                2.0 * w[i,j] * ( (F[i,0,2]+G[j,0,2]) * (F[i,2,1]+G[j,2,1]) 
                    - (F[i,2,2]+G[j,2,2]) * (F[i,0,1]+G[j,0,1]) ) / det
            out[j,0,2] += \
                2.0 * w[i,j] * ( (F[i,0,1]+G[j,0,1]) * (F[i,1,2]+G[j,1,2]) 
                    - (F[i,0,2]+G[j,0,2]) * (F[i,1,1]+G[j,1,1]) ) / det
            out[j,1,0] += \
                2.0 * w[i,j] * ( (F[i,1,2]+G[j,1,2]) * (F[i,2,0]+G[j,2,0]) 
                    - (F[i,1,0]+G[j,1,0]) * (F[i,2,2]+G[j,2,2]) ) / det
            out[j,1,1] += \
                2.0 * w[i,j] * ( (F[i,0,0]+G[j,0,0]) * (F[i,2,2]+G[j,2,2]) 
                    - (F[i,0,2]+G[j,0,2]) * (F[i,2,0]+G[j,2,0]) ) / det
            out[j,1,2] += \
                2.0 * w[i,j] * ( (F[i,0,2]+G[j,0,2]) * (F[i,1,0]+G[j,1,0]) 
                    - (F[i,0,0]+G[j,0,0]) * (F[i,1,2]+G[j,1,2]) ) / det
            out[j,2,0] += \
                2.0 * w[i,j] * ( (F[i,1,0]+G[j,1,0]) * (F[i,2,1]+G[j,2,1]) 
                    - (F[i,1,1]+G[j,1,1]) * (F[i,2,0]+G[j,2,0]) ) / det
            out[j,2,1] += \
                2.0 * w[i,j] * ( (F[i,0,1]+G[j,0,1]) * (F[i,2,0]+G[j,2,0]) 
                    - (F[i,0,0]+G[j,0,0]) * (F[i,2,1]+G[j,2,1]) ) / det
            out[j,2,2] += \
                2.0 * w[i,j] * ( (F[i,0,0]+G[j,0,0]) * (F[i,1,1]+G[j,1,1]) 
                    - (F[i,0,1]+G[j,0,1]) * (F[i,1,0]+G[j,1,0]) ) / det
    return np.asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
def riccatiG_inv(cnp.ndarray[double, ndim=2] q, 
                 cnp.ndarray[double, ndim=3] F, 
                 cnp.ndarray[double, ndim=3] G, 
                 cnp.ndarray[double, ndim=2] t_opt, 
                 cnp.ndarray[double, ndim=3] RB):
    """Computes GG[j] = sum_i q[i,j] * (R[i,j].T*F[i]*R[i,j]+G[j]).I ."""
    
    cdef int m = q.shape[0], n = q.shape[1]
    cdef int d = F.shape[1], f = (RB.shape[0]-1)//2
    cdef cnp.ndarray[double, ndim=3] GG = np.zeros((n,d,d))
    cdef int tid, num_threads = omp_get_max_threads()
    cdef int memory_per_thread = 3*d*d
    cdef double* work = \
        <double*> malloc(sizeof(double) * num_threads * memory_per_thread)
    cdef double* R
    cdef double* tmp
    cdef double* FRpG # FRpG[i,j] = R[i,j].T * F[i] * R[i,j] + G[j]
    cdef double c, s
    cdef int i, j, k, l, r, info
    
    with nogil:
        for j in prange(n, schedule='static'):
            tid = omp_get_thread_num()
            R = &work[memory_per_thread * tid]
            tmp = &work[memory_per_thread * tid + d*d]
            FRpG = &work[memory_per_thread * tid + 2*d*d]
            
            for i in range(m):
                # compute rotation matrix:
                for k in range(d):
                    for l in range(d):
                        R[k+d*l] = RB[0,k,l]
                for r in range(f):
                    c = cos((r+1)*t_opt[i,j])
                    s = sin((r+1)*t_opt[i,j])
                    for k in range(d):
                        for l in range(d):
                            R[k+d*l] += c * RB[2*r+1,k,l] + s * RB[2*r+2,k,l]
                
                # compute R^T * F * R + G:
                for k in range(d):
                    for l in range(d):
                        tmp[d*k+l] = 0
                        for r in range(d):
                            tmp[d*k+l] += R[d*k+r] * F[i,r,l]
                for k in range(d):
                    for l in range(k,d):
                        FRpG[d*k+l] = G[j,k,l]
                        for r in range(d):
                            FRpG[d*k+l] += tmp[d*k+r] * R[d*l+r]
                
                # compute inverse:
                dpotrf('L', &d, FRpG, &d, &info)
                dpotri('L', &d, FRpG, &d, &info)
                
                # add inverse to the output:
                for k in range(d):
                    for l in range(k,d):
                        GG[j,k,l] += 2. * q[i,j] * FRpG[d*k+l]
        
            # symmetrize result:
            for k in range(1,d):
                for l in range(k):
                    GG[j,k,l] = GG[j,l,k]
                    
    free(work)
    return GG

