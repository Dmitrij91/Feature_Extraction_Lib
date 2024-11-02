import numpy as np
from scipy.sparse import csr_matrix
from cython.parallel import prange
cimport numpy as np
from scipy.linalg.cython_lapack cimport dpotrf
cimport cython
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.stdlib cimport malloc, free
from libc.math cimport log
    


@cython.boundscheck(False)
@cython.wraparound(False)
def distancematrix_stein(double[:,:,:] data, double[:,:,:] prototype, 
                         double[:,:] out = None):
    assert data.shape[1] == data.shape[2]
    assert prototype.shape[1] == prototype.shape[2]
    assert data.shape[1] == prototype.shape[1]
    cdef int s = data.shape[1], m = data.shape[0], n = prototype.shape[0]
    cdef int num_t = omp_get_max_threads(), tid
    cdef double[:,:] dist
    if out is not None:
        dist = out
    else:
        dist = np.empty((m,n))
    cdef double* work = <double*> malloc(sizeof(double) * num_t * s * s)
    cdef double* logdet_d = <double*> malloc(sizeof(double) * m)
    cdef double* logdet_p = <double*> malloc(sizeof(double) * n)
    cdef int i, j, k, l, info
    with nogil:
        # Compute logdet_d (= 0.5 * logdet of data):
        for i in prange(m):
            tid = omp_get_thread_num()
            # copy upper triangular part into working space:
            for j in range(s):
                for k in range(j,s):
                    work[s*s*tid + s*j + k] = data[i,j,k]
            # Cholesky decomposition:
            dpotrf('L', &s, work + s*s*tid, &s, &info)
            logdet_d[i] = 1.0
            for j in range(s):
                logdet_d[i] *= work[s*s*tid + (s+1)*j]
            logdet_d[i] = log(logdet_d[i])
        # Compute logdet_p (= 0.5 * logdet of prototype):
        for i in prange(n):
            tid = omp_get_thread_num()
            # Copy upper triangular part into work space:
            for j in range(s):
                for k in range(j,s):
                    work[s*s*tid + s*j+k] = prototype[i,j,k]
            # Cholesky decomposition:
            dpotrf('L', &s, work + s*s*tid, &s, &info)
            logdet_p[i] = 1.0
            for j in range(s):
                logdet_p[i] *= work[s*s*tid + (s+1)*j]
            logdet_p[i] = log(logdet_p[i])
        # Finally compute distance matrix:
        for i in prange(m):
            tid = omp_get_thread_num()
            for j in range(n):
                # Copy upper triangular part of (data[i] + prototype[j])/2 
                # into working space:
                for k in range(s):
                    for l in range(k,s):
                        work[s*s*tid + s*k + l] = \
                            0.5 * (data[i,k,l] + prototype[j,k,l])
                # Cholesky decomposition:
                dpotrf('L', &s, work + s*s*tid, &s, &info)
                dist[i,j] = 1.0
                for k in range(s):
                    dist[i,j] *= work[s*s*tid + s*k + k]
                dist[i,j] = 2.0 * log(dist[i,j]) - logdet_d[i] - logdet_p[j]
    free(work)
    free(logdet_d)
    free(logdet_p)
    return np.asarray(dist)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def normalize_adj(np.ndarray[double, ndim=1] data, int[:] indptr):
    # Normalize the adj matrix such that weights sum to 1.
    cdef int m = data.shape[0], n = indptr.shape[0]
    assert indptr[n-1] == m
    cdef int i, j
    cdef double s
    for i in range(n-1):
        s = 0.0
        for j in range(indptr[i], indptr[i+1]):
            s += data[j]
        for j in range(indptr[i], indptr[i+1]):
            data[j] /= s
    return data



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def adj_matrix_uniform(int m, int n, double[:,:] w):
    # Returns the matrix A such that A*vec(I) = correlate(I,w) 
    # for (m,n) images I.
    assert w.shape[0]%2 == 1 and w.shape[1]%2 == 1
    assert m > 0 and n > 0
    cdef int px = w.shape[0], py = w.shape[1]
    cdef int px2 = (px-1)//2, py2 = (py-1)//2
    cdef int num_entries = ((m-px+1)*px+(3*px-1)*(px-1)//4) \
                           * ((n-py+1)*py+(3*py-1)*(py-1)//4)
    cdef np.ndarray[double, ndim=1, negative_indices=False] data = \
        np.empty((num_entries,))
    cdef np.ndarray[int, ndim=1, negative_indices=False] indices = \
        np.empty((num_entries,), dtype=np.intc)
    cdef np.ndarray[int, ndim=1, negative_indices=False] indptr = \
        np.empty((m*n+1,), dtype=np.intc)
    cdef int i, j, k, l, a, b, ind_px, ind=0
    # Compute data of avrMatrix:
    indptr[0] = 0
    for i in range(m):
        for j in range(n):
            ind_px = 0
            for k in range(max(0, i-px2), min(m, i+px2+1)):
                for l in range(max(0, j-py2), min(n, j+py2+1)):
                    data[ind+ind_px] = w[k-i+px2, l-j+py2]
                    indices[ind+ind_px] = n*k+l
                    ind_px += 1
            ind += ind_px
            indptr[n*i+j+1] = ind
    #assert ind == num_entries
    return data, indices, indptr



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def adj_matrix_nonuniform(double[:,:,:,:] w):
    assert w.shape[2]%2 == 1 and w.shape[3]%2 == 1
    cdef int m = w.shape[0], n = w.shape[1]
    cdef int px = w.shape[2], py = w.shape[3]
    cdef int px2 = (px-1)//2, py2 = (py-1)//2
    cdef int num_entries = ((m-px+1)*px+(3*px-1)*(px-1)//4) \
                           * ((n-py+1)*py+(3*py-1)*(py-1)//4)
    cdef np.ndarray[double, ndim=1, negative_indices=False] data = \
        np.empty((num_entries,))
    cdef np.ndarray[int, ndim=1, negative_indices=False] indices = \
        np.empty((num_entries,), dtype=np.intc)
    cdef np.ndarray[int, ndim=1, negative_indices=False] indptr = \
        np.empty((m*n+1,), dtype=np.intc)
    cdef int i, j, k, l, ind_px, ind = 0
    
    indptr[0] = 0
    for i in range(m):
        for j in range(n):
            ind_px = 0
            for k in range(px):
                if not 0 <= i + k - px2 < m:
                    continue
                for l in range(py):
                    if not 0 <= j + l - py2 < n:
                        continue
                    data[ind + ind_px] = w[i,j,k,l]
                    indices[ind + ind_px] = n * (i + k - px2) + (j + l - py2)
                    ind_px += 1
            ind += ind_px
            indptr[n*i+j+1] = ind
    #assert ind == num_entries       
    return data, indices, indptr



@cython.wraparound(False)
@cython.boundscheck(False)
def adj_mul_distance(np.ndarray[double, ndim=1] data, int[:] indices, 
                     int[:] indptr, double[:,:,:] F):
    assert F.shape[1] == F.shape[2]
    assert data.shape[0] == indices.shape[0]
    assert indptr.shape[0] == F.shape[0] + 1
    assert indptr[F.shape[0]] == data.shape[0]
    cdef int m = F.shape[0], s = F.shape[1]
    cdef int num_t = omp_get_max_threads(), tid
    cdef double* work = <double*> malloc(sizeof(double) * num_t * s * s)
    cdef double* d = <double*> malloc(sizeof(double) * num_t)
    cdef int i, j, k, l, info
    for i in prange(m, nogil=True):
        tid = omp_get_thread_num()
        for j in range(indptr[i], indptr[i+1]):
            d[tid] = 0.0
            # log det( 0.5*(F_i + F_j)):
            for k in range(s):
                for l in range(k,s):
                    work[s*s*tid + s*k + l] = 0.5 * (F[indices[j], k, l] +
                                                     F[i, k, l])
            dpotrf('L', &s, work + s*s*tid, &s, &info)
            for k in range(s):
                d[tid] += log(work[s*s*tid + (s+1)*k])
            # log det( F_j ):
            for k in range(s):
                for l in range(k,s):
                    work[s*s*tid + s*k + l] = F[indices[j], k, l]
            dpotrf('L', &s, work + s*s*tid, &s, &info)
            for k in range(s):
                d[tid] -= 0.5*log(work[s*s*tid + (s+1)*k])
            # log det( F_i ):
            for k in range(s):
                for l in range(k,s):
                    work[s*s*tid + s*k + l] = F[i, k, l]
            dpotrf('L', &s, work + s*s*tid, &s, &info)
            for k in range(s):
                d[tid] -= 0.5*log(work[s*s*tid + (s+1)*k])
            data[j] *= d[tid]
    free(work)
    free(d)
    return data
    