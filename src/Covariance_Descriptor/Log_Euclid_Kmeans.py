import numpy as np
import os
import sys

try:
    import cupy as cp
except ImportError:
    print('Module not found: cupy')

# 
def kmeans(X, K=10, num_init=1, max_iter=999, threads_per_block=1024,
           verbose=True):
    """Computes the k-means clustering using k-center as initialization.

    Input
    -----
    X : (N,d) array
        N points in the d-dim. Euclidean space
    K : int
        Number of clusters
    num_init : int
        Number of initializations (with greedy k-center)
    max_iter : int
        Maximal number of k-means iterations
    threads_per_block : int
        Number of threads per block for the computation on the CUDA card.
        Should be approx. the square root of N and divisible by 32.
        It can be maximal 1024.
    verbose : bool
        If False, console prints are suppressed.

    Output
    ------
    C : (K,d) array
        Cluster centers
    cl : (N,) int array
        Assignment of centers, i.e., cl[i]=j if center C[j] is assigned to X[i]
    """

    # sanity checks:
    assert X.ndim == 2
    assert isinstance(K, int) and 1 <= K <= 1024
    assert isinstance(num_init, int) and num_init >= 1
    assert isinstance(max_iter, int) and max_iter >= 0
    assert isinstance(threads_per_block, int) and 1 <= threads_per_block <= 1024
    assert isinstance(verbose, bool)

    if verbose:
        output = sys.stdout
    else:
        output = open(os.devnull, 'w')

    # DEFINE CUDA FUNCTIONS: dist2, assign_NN, compute_centers and
    #     compute_objective_parts
    #
    # dist2(X,y) computes the (squared) Euclidean distance between
    # the (N,d) array X and the (d,) vector y.
    # Output: (N,) vector D
    dist2 = cp.RawKernel(r'''
        extern "C" __global__
        void dist2(const float* x, float* y, float* D, int N, int d) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < N) {
                float tmp = 0.;
                for (int j=0; j<d; ++j) {
                    tmp += (x[d*i+j] - y[j]) * (x[d*i+j] - y[j]);
                }
                D[i] = min(D[i], tmp);
            }
        }
        ''', 'dist2')

    # assign_NN(X,C) computes the nearest neighbor assignment of centers given
    # by the (K,d) array C to the points given by the (N,d) array X:
    # Output: (N,) int array (indices)
    assign_NN = cp.RawKernel(r'''
        extern "C" __global__
        void assign(const float* x, const float* c, int* indices, 
                    int N, int K, int d) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int idx = -1;
            float min_dist = __int_as_float(0x7f800000); // = inf
            float dist = 0.;
        
            if (i < N) {
                for (int k=0; k<K; ++k) {
                    dist = 0.;
                    for (int j=0; j<d; ++j) {
                        dist += (x[d*i+j] - c[d*k+j]) * (x[d*i+j] - c[d*k+j]);
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        idx = k;
                    }
                }
                indices[i] = idx;
            }
        }
        ''', 'assign')

    # compute_centers(X, C, idx) computes the cluster centers for the
    # assignments given by (N,) int array idx and saves the centers in
    # the (K,d) array C.
    # It is a naive implementation, but it is at least memory efficient.
    compute_centers = cp.RawKernel(r'''
        extern "C" __global__
        void average(const float* x, float* c, int* indices, 
                     int N, int K, int d) {
            int k = threadIdx.x;
            int j = blockIdx.x;
            float w = 0.;
            c[k*d+j] = 0.;
            for (int i=0; i<N; ++i) {
                if (indices[i] == k) {
                    c[k*d+j] += x[i*d+j];
                    w += 1.;
                }
            }
            c[k*d+j] /= w;
        }
        ''', 'average')

    # compute_objective_parts(X, C, idx) computes for each point
    # the squared distance to its cluster center.
    # Output: (N,) array
    compute_objective_parts = cp.RawKernel(r'''
        extern "C" __global__
        void objective(const float* x, const float* c, const int* indices, 
                       float* out, int N, int K, int d) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < N) {
                float sum = 0.;
                float tmp = 0.;
                for (int j=0; j<d; ++j) {
                    tmp = x[d*i+j] - c[d*indices[i]+j];
                    sum += tmp * tmp;
                }
                out[i] = sum;
            }
        }
        ''', 'objective')

    # some auxiliary variable:
    N, d = X.shape
    X_gpu = cp.array(X, dtype='float32')
    C_gpu = cp.empty((K, d), dtype='float32')
    J_opt = float('inf')

    # set parameter for CUDA kernel:
    num_blocks = int(np.ceil(N / threads_per_block))
    grid = (num_blocks,)
    block = (threads_per_block,)

    # reserve temp memory on gpu:
    D_tmp = cp.empty((N,), dtype='float32')
    J_parts = cp.empty((N,), dtype='float32')
    indices_old = cp.full((N,), -1, dtype='int32')
    indices = cp.empty((N,), dtype='int32')

    for init in range(num_init):
        print('### start %d. initialization ###' % (init + 1), file=output)
        # greedy k-center clustering:
        n = np.random.randint(N)
        D_tmp[:] = float('inf')
        C_gpu[0, :] = X_gpu[n, :]
        for k in range(1, K):
            dist2(grid, block, (X_gpu, C_gpu[k - 1, :], D_tmp, N, d))
            j = cp.argmax(D_tmp)
            C_gpu[k, :] = X_gpu[j, :]

        # k-means iteration:
        assign_NN(grid, block, (X_gpu, C_gpu, indices, N, K, d))
        for it in range(max_iter):
            num_changes = cp.count_nonzero(cp.not_equal(indices, indices_old))
            print('%d. iteration: %d assignments changed' % (it + 1, num_changes),
                  file=output)
            if num_changes == 0:
                print('   --> stop', file=output)
                break
            indices_old[:] = indices
            compute_centers((d,), (K,), (X_gpu, C_gpu, indices, N, K, d))
            assign_NN(grid, block, (X_gpu, C_gpu, indices, N, K, d))

        # compare objective:
        compute_objective_parts(grid, block,
                                (X_gpu, C_gpu, indices, J_parts, N, K, d))
        J = cp.sum(J_parts)
        if J < J_opt:
            C = C_gpu.get()
            cl = indices.get()
            J_opt = J
            print('objective = %g   -->   New prototypes selected!' % J,
                  file=output)
        else:
            print('objective = %g' % J, file=output)
        print('', file=output)
    print('best objective value: %g' % J_opt, file=output)
    return C, cl
