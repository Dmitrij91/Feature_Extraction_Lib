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
@cython.cdivision(True)
def distance_stein_inv(cnp.ndarray[double, ndim=3] F, 
                       cnp.ndarray[double, ndim=3] G, 
                       cnp.ndarray[double, ndim=3] RB,
                       int num_points, int num_angles):
    """Computes the distance matrix for the rotation-inv. Stein divergence."""
    
    # declarations and initializations:
    #
    cdef int m = F.shape[0], n = G.shape[0]
    cdef int d = F.shape[1], f = (RB.shape[0]-1)//2
    cdef int i, j, k, l, r
    cdef cnp.ndarray[double, ndim=2] D = np.empty((m,n))
    cdef cnp.ndarray[double, ndim=2] t_opt = np.zeros((m,n))
    cdef double* logdet_F = <double*> malloc(sizeof(double) * m)
    cdef double* logdet_G = <double*> malloc(sizeof(double) * n)
    cdef double omega = pi / num_points
    cdef int num_samples = 2 * num_points
    
    # declarations of variables for LAPACK methods:
    #
    cdef int M = 6*num_points, N = 6*num_points, deg = 3*num_points, info = 0
    cdef double* work
    cdef double one = 1., zero = 0.
    
    
    # Initialize lhs matrix for lstsq of Fourier polynomial 
    # a_{0} + sum_j a_{j}*cos(jt) + b_{j}*sin(jt).
    # The matrix is of size (3*num_samples)x(2*deg) and
    # each row contains the 'coefficients' of 
    # [a_{d},a_{d-1},...,a_{0},b_{1},...,b_{d-1}]
    # The first num_samples rows describe the values of the polynomial,
    # the next num_samples rows describe the first derivatives, etc.
    # A_lstsq save the transpose of the matrix, so 
    # LQ = A^T --> A = Q^T * L^T --> A^{-1} = L^{-T} * Q
    cdef double* A_lstsq = <double*> malloc(sizeof(double) * M * N)
    cdef double* t_samples = <double*> malloc(sizeof(double) * num_samples)
    for k in range(num_samples):
        A_lstsq[deg+N*k] = 1.
        A_lstsq[deg+N*(k+num_samples)] = 0.
        A_lstsq[deg+N*(k+2*num_samples)] = 0.
        t_samples[k] = k * omega
        A_lstsq[N*k] = cos( deg * t_samples[k] ) # = cos(Nt)
        A_lstsq[N*(k+num_samples)] = \
            -deg * sin( deg * t_samples[k] ) # = -N * sin(Nt)
        A_lstsq[N*(k+2*num_samples)] = \
            -deg*deg * A_lstsq[N*k]
        for l in range(1,deg):
            A_lstsq[l+N*k] = \
                cos( (deg-l) * t_samples[k] ) # = cos(jt)
            A_lstsq[N-l+N*k] = \
                sin( (deg-l) * t_samples[k] ) # = sin(jt)
            A_lstsq[l+N*(k+num_samples)] = \
                -(deg-l) * A_lstsq[N-l+N*k] # = -j sin(jt)
            A_lstsq[N-l+N*(k+num_samples)] = \
                (deg-l) * A_lstsq[l+N*k] # = j cos(jt)
            A_lstsq[l+N*(k+2*num_samples)] = \
                -(deg-l)*(deg-l) * A_lstsq[l+N*k] # = -j^2 cos(jt)
            A_lstsq[N-l+N*(k+2*num_samples)] = \
                -(deg-l)*(deg-l) * A_lstsq[N-l+N*k]  # = -j^2 sin(jt)
    
    
    # compute (transposed) pseudo-inverse of A_lstsq (-> A_lstsq_I):
    #
    cdef double* A_lstsq_I = <double*> malloc(sizeof(double) * M * N)
    # scalar factors of reflectors (of Q):
    cdef double* tau = <double*> malloc(sizeof(double) * N)

    # compute LQ decomposition of A_lstsq_T:
    work = <double*> malloc(sizeof(double) * N)
    dgelqf(&N, &M, A_lstsq, &N, tau, work, &N, &info)
    assert info == 0
    
    # explicitly compute matrix Q (save in A_lstsq_I):
    memcpy(A_lstsq_I, A_lstsq, sizeof(double) * M * N)
    dorglq(&N, &M, &N, A_lstsq_I, &N, tau, work, &N, &info)
    assert info == 0
    
    # compute A_lstsq_I = L^{-T} * Q:
    dtrsm('L', 'L', 'T', 'N', &N, &M, &one, A_lstsq, &N, A_lstsq_I, &N)
    
    # transpose A_lstsq_I (for minimal speed boost):
    memcpy(A_lstsq, A_lstsq_I, sizeof(double) * M * N)
    for k in range(M):
        for l in range(N):
            A_lstsq_I[k+M*l] = A_lstsq[l+N*k]
    
    # tau and A_lstsq are not needed any more:
    free(work)
    free(tau)
    free(A_lstsq)    
    
    
    # precompute recurring entities:
    #
    cdef cnp.ndarray[double, ndim=2] Omega = np.zeros((d,d)) # R = exp(t*Omega)
    for k in range(f):
        Omega += (k+1) * RB[2*k+2,:,:]
    
    # compute rotation matrix for sample angles:
    cdef cnp.ndarray[double, ndim=3] R_samples = \
        np.empty((num_samples,d,d), dtype=RB.dtype)
    for k in range(num_samples):
        R_samples[k,:,:] = RB[0,:,:]
        for l in range(f):
            R_samples[k,:,:] += cos((l+1)*t_samples[k]) * RB[2*l+1,:,:]
            R_samples[k,:,:] += sin((l+1)*t_samples[k]) * RB[2*l+2,:,:]
    
    # GR[n,r] = R[r] * G[n] * R[r].T
    cdef cnp.ndarray[double, ndim=4] GR = \
        np.einsum('rij,njk,rlk->nril', R_samples, G, R_samples)
    
    # FO[m] = F[m] * Omega - Omega * F[m]
    cdef cnp.ndarray[double, ndim=3] FO = np.tensordot(F, Omega, axes=(2,0))
    FO += np.transpose(FO, (0,2,1))
    
    # GRO[n,r] = GR[n,r] * Omega - Omega * GR[n,r]
    cdef cnp.ndarray[double, ndim=4] GRO = np.tensordot(GR, Omega, axes=(3,0))
    GRO += np.transpose(GRO, (0,1,3,2))
    
    cdef double dlog2 = d * log(2.) # logdet(X/2) = logdet(X) - d * log(2.)
    
    omega = 2. * pi / num_angles
    cdef double* t = <double*> malloc(sizeof(double) * num_angles)
    # A_polynomial^T * coeffs = polynomial values
    cdef double* A_polynomial = \
        <double*> malloc(sizeof(double) * num_angles * N)
    cdef double* A_deriv1 = <double*> malloc(sizeof(double) * num_angles * N)
    cdef double* A_deriv2 = <double*> malloc(sizeof(double) * num_angles * N)
    for k in range(num_angles):
        t[k] = k * omega
        A_polynomial[N*k+deg] = 1.
        A_deriv1[N*k+deg] = 0.
        A_deriv2[N*k+deg] = 0.
        A_polynomial[N*k] = cos( deg * t[k] )
        A_deriv1[N*k] = -deg * sin( deg * t[k] )
        A_deriv2[N*k] = -deg * deg * A_polynomial[N*k]
        for l in range(1,deg):
            A_polynomial[N*k+l] = cos( (deg-l) * t[k] )
            A_polynomial[N*k+N-l] = sin( (deg-l) * t[k])
            A_deriv1[N*k+l] = -(deg-l) * A_polynomial[N*k+N-l]
            A_deriv1[N*k+N-l] = (deg-l) * A_polynomial[N*k+l]
            A_deriv2[N*k+l] = - (deg-l) * A_deriv1[N*k+N-l]
            A_deriv2[N*k+N-l] = (deg-l) * A_deriv1[N*k+l]
            
    # variables for openmp / memory management:
    cdef int num_threads = omp_get_max_threads()
    cdef int tid # = thread id
    cdef int memory_per_thread = 5*d*d + M + N + num_angles
    cdef int memory_offset[7] # offset for working spaces defined below
    # 0: FpGR (size = d*d)
    # 1: FOaGRO (size = d*2d)
    # 2: R_opt (size = d*d)
    # 3: tmp_matrix (size = d*d)
    # 4: b_lstsq (size = M)
    # 5: coeffs (size = N)
    # 6: val_polynomial (size = num_angles)
    memory_offset[:] = [0, d*d, 3*d*d, 4*d*d, 5*d*d, 5*d*d+M, 5*d*d+M+N]
    
    # preallocate enough memory for all temporary working spaces:
    work = <double*> malloc(sizeof(double) * num_threads * memory_per_thread)
    
    
    # declaration of temporary working spaces / variables:
    #
    cdef double* X # for storing a matrix
    cdef double* FpGR # for storing  F[i] + GR[j,r]
    cdef double* FOaGRO # for storing ( FO[i], GRO[j,r] )
    cdef double* R_opt # = optimal rotation
    cdef double* tmp_matrix
    cdef double* b_lstsq # = rhs of lstsq problem for computing coeffs
    cdef double* coeffs # = coeffs of Fourier polynomial
    cdef double* val_polynomial # = values of Fourier polynomial
    
    cdef double const_D # part of D_S(X,RYR^T) which does not depend on R
    cdef double min_approx, deriv1, deriv2 # value + derivatives of polynomial
    cdef double c, s # = cos(t), sin(t)
    cdef int argmin
    
    # constants for LAPACK calls:
    cdef int d2 = d*2, one_int = 1
    
    
    # main part:
    #
    with nogil:
        #compute log(det(F)):
        for i in prange(m, schedule='static'):
            tid = omp_get_thread_num()
            X = &work[memory_per_thread*tid]
            # copy upper triangular part into working space:
            for k in range(d):
                for l in range(k,d):
                    X[d*k+l] = F[i,k,l]
            logdet_F[i] = logdet(X, d, &info)
        #compute log(det(G)):
        for j in prange(n, schedule='static'):
            tid = omp_get_thread_num()
            X = &work[memory_per_thread*tid]
            # copy upper triangular part into working space:
            for k in range(d):
                for l in range(k,d):
                    X[d*k+l] = G[j,k,l]
            logdet_G[j] = logdet(X, d, &info)
        # main iteration:
        for i in prange(m, schedule='static'):
            # distribute allocated memory to temporary working spaces:
            tid = omp_get_thread_num()
            FpGR = &work[memory_per_thread * tid + memory_offset[0]]
            FOaGRO = &work[memory_per_thread * tid + memory_offset[1]]
            R_opt = &work[memory_per_thread * tid + memory_offset[2]]
            tmp_matrix = &work[memory_per_thread * tid + memory_offset[3]]
            b_lstsq = &work[memory_per_thread * tid + memory_offset[4]]
            coeffs = &work[memory_per_thread * tid + memory_offset[5]]
            val_polynomial = &work[memory_per_thread * tid + memory_offset[6]]
            
            for j in range(n):
                const_D = dlog2 + 0.5 * logdet_F[i] + 0.5 * logdet_G[j]
                for r in range(num_samples):
                    for k in range(d):
                        for l in range(k,d):
                            FpGR[d*k+l] = F[i,k,l] + GR[j,r,k,l]
                    b_lstsq[r] = logdet(FpGR, d, &info) - const_D
                    # Now, FpGR stores the choleski decomp of F[i]+GR[j,r]
                    for l in range(d):
                        for k in range(d):
                            FOaGRO[k+d*l] = FO[i,k,l]
                    for l in range(d):
                        for k in range(d):
                            FOaGRO[k+d*l+d*d] = GRO[j,r,k,l]
                    # compute (F[i]+GR[j,r])^{-1} * [FO[i], GRO[j,r]]
                    # into FOaGRO:
                    dpotrs('L', &d, &d2, FpGR, &d, FOaGRO, &d, &info)
                    # compute first derivative, i.e. 
                    # trace( (F[i]+GR[j,r])^{-1} * FO[i] ) :
                    b_lstsq[r+num_samples] = 0.0
                    for k in range(d):
                        b_lstsq[r+num_samples] += FOaGRO[k*(d+1)]
                    # compute second derivative, i.e. 
                    # tr( (F[i]+GR[j,r])^{-1} * FO[i] 
                    #     * (F[i]+GR[j,r])^{-1} * GRO[j,r] ):
                    b_lstsq[r+2*num_samples] = 0.0
                    for l in range(d):
                        for k in range(d):
                            b_lstsq[r+2*num_samples] += \
                                FOaGRO[k+d*l] * FOaGRO[l+d*k+d*d]
                # compute coeffs:
                dgemv('T', &M, &N, &one, A_lstsq_I, &M, b_lstsq, &one_int, 
                      &zero, coeffs, &one_int)
                # compute polynomial:
                dgemv('T', &N, &num_angles, &one, A_polynomial, &N, 
                      coeffs, &one_int, &zero, val_polynomial, &one_int)
                
                # brute force search for minimum of polynomial:
                min_approx = val_polynomial[0]
                argmin = 0
                for k in range(1,num_angles):
                    if val_polynomial[k] < min_approx:
                        min_approx = val_polynomial[k]
                        argmin = k
                
                # do one Newton step:
                deriv1 = ddot(&N, &A_deriv1[N*argmin], &one_int, 
                              coeffs, &one_int)
                deriv2 = ddot(&N, &A_deriv2[N*argmin], &one_int, 
                              coeffs, &one_int)
                t_opt[i,j] = t[argmin] - deriv1 / deriv2
                
                
                # compute 'optimal' rotation matrix: 
                for k in range(d):
                    for l in range(d):
                        R_opt[d*k+l] = RB[0,k,l]
                for r in range(f):
                    c = cos((r+1)*t_opt[i,j])
                    s = sin((r+1)*t_opt[i,j])
                    for k in range(d):
                        for l in range(d):
                            R_opt[d*k+l] += c * RB[2*r+1,k,l]
                            R_opt[d*k+l] += s * RB[2*r+2,k,l]
                
                # compute 'optimal'  F[i] + R*G[j]*R^T :
                for k in range(d):
                    for l in range(d):
                        tmp_matrix[d*k+l] = 0.0
                        for r in range(d):
                            tmp_matrix[d*k+l] += R_opt[d*k+r] * G[j,r,l]
                for k in range(d):
                    for l in range(k,d):
                        FpGR[d*k+l] = F[i,k,l]
                        for r in range(d):
                            FpGR[d*k+l] += tmp_matrix[d*k+r] * R_opt[d*l+r]
                
                # compute 'optimal' distance
                D[i,j] = logdet(FpGR, d, &info) - const_D
                
                # check of any computed distance is smaller:
                for k in range(num_samples):
                    if b_lstsq[k] < D[i,j]:
                        D[i,j] = b_lstsq[k]
                        t_opt[i,j] = t_samples[k]
                        
    
    # free allocated memory:
    #
    free(logdet_F)
    free(logdet_G)
    free(A_lstsq_I)
    free(t_samples)
    free(t)
    free(A_polynomial)
    free(A_deriv1)
    free(A_deriv2)
    free(work)
    
    t_opt = t_opt % (2.*pi)
    
    return D, t_opt



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

