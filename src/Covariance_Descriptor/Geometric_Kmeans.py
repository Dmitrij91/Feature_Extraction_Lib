
from sklearn.cluster import k_means as kmeans
import Descriptor_Cython
import Descriptor_Cython.cython_distance
import Descriptor_Cython.Geometric_Mean_Utils
import linalgh
import numpy as np
import time
from scipy.linalg import sqrtm,logm,expm


def K_means(f_pd,K,Max_Iter = 100,n_init_kmeans=10,Method = 'Log_Euclid',Metric = 'Riemmann'):
    s = f_pd.shape[2]
    N_Pixel = f_pd.shape[0]
    Random_Indices = np.random.choice(N_Pixel,K)
    # Initialization of K random Centroids
    Clusters = f_pd[Random_Indices]
    D = np.zeros((N_Pixel,K))
    W = np.zeros((N_Pixel,K))
    W_old = np.zeros((N_Pixel,K))
    tol_mean = 1e-4
    if Method == 'Log_Euclid': 
        
        '''Cumpute Clusters 
        Initialize G by log-Euclidean kmeans:
        Transfer the Frobenius inner product to Euclidian'''
        
        A = matrix_sym2vec(s)
        
        '''Matrix_Log,such that |X|_2 = |log f_pd|_F, == vec(log_f_pd)'''
        
        X = linalgh.logmh(f_pd).reshape((N_Pixel,s*s)).dot(A.T)
        print(X)
        # K_means on Eucliadian space == Tangential space log f_pd
        X = kmeans(X, K, n_init=n_init_kmeans)[0]
        # Transform to Manifold by exp such that G = exp(log X_1),...exp(log_X_K)
        A = A.T.dot(linalgh.invh(A.dot(A.T)))
        Clusters = linalgh.expmh(np.einsum('kl,jl->jk', A, X).reshape((K,s,s))) 
        if Metric == 'Log_Det':
            for it in range(Max_Iter):
                D = Descriptor_Cython.cython_distance.distance_stein(f_pd, Clusters)
                W[range(N_Pixel),np.argmin(D, axis=-1)] = 1.0 # assignment by NN
                if np.allclose(W_old, W):
                    break
                for j in range(Max_Iter):
                    mcG = Descriptor_Cython.cython_distance.riccatiG(W/np.sum(W, axis=0), f_pd, Clusters)
                    grad = Clusters[0] - Clusters[0].dot(mcG[0]).dot(Clusters[0])
                    grad = np.sqrt(np.mean(np.abs(grad) ** 2))
                    Clusters = inv_hpd(mcG)
                    if grad < tol_mean:
                        break
                W_old = W
        elif Metric == 'Riemmann':
            for it in range(Max_Iter):
                D = Descriptor_Cython.Geometric_Mean_Utils.distance_Riemmann(f_pd, Clusters)
                W[range(N_Pixel),np.argmin(D, axis=-1)] = 1.0 # assignment by NN
                if np.allclose(W_old, W):
                    break
                Clusters = Descriptor_Cython.Geometric_Mean_Utils.Riemann_Update_Cluster(W, f_pd, Clusters)
                W_old = W
    return Clusters.reshape(-1,s*s)

def greedy_clustering(X, K, metric = 'Stein', init = None, out_dist = False):
    """
    Computes a metric clustering by using a greedy picking of cluster centers.
    
    Parameters
    ----------
    X : array
        An array containing the N data points. If metric == 'Stein, then X 
        should be an (N,s,s) array. If metric == 'Euclidean', then X should 
        be an (N,s) array.
    K : int
        Number of clusters.
    metric : str or callable
        Distance function. It is either a callable function or a string for a 
        predefined metric ('Stein' or 'Euclidean')
    init : array or None
        Initialization point for the greedy clustering (first cluster center).
        If init is None, clustering is initialized by choosing the nearest 
        point of the average.
    out_dist : bool
        If out_dist == True, the distances dist (see below).
    
    Returns
    -------
    Y : array
        An array containing the K cluster centers. If metric == 'Stein', then Y 
        is an (K,s,s) array. If metric == 'Euclidean', then Y is an 
        (K,s) array.
    dist : array (optional)
        An (K,) array with dist[i] = D(X, Y[:i]).
    """
    
    # sanity checks:
    assert isinstance(X, np.ndarray)
    assert isinstance(K, int) and K > 0
    assert isinstance(metric, str) or callable(metric)
    assert init is None or (isinstance(init, np.ndarray) 
                            and init.shape == X.shape[1:])
    if isinstance(metric, str):
        assert metric in ['Stein', 'Euclidean']
        if metric == 'Stein':
            assert X.ndim == 3
            assert X.shape[1] == X.shape[2]
        elif metric == 'Euclidean':
            assert X.ndim == 2
    
    N = X.shape[0]
    Y = np.empty((K,*X.shape[1:]))
    mdist = np.empty((K,))
    
    # initialization:
    if metric == 'Stein':
        Y[0] = expmh(np.mean(logmh(X), axis=0))
        dist = ctools.distancematrix_stein
    elif metric == 'Euclidean':
        Y[0] = np.mean(X, axis=0)
        dist = cdist
    elif callable(metric):
        Y[0] = X[np.random.randint(N)]
        dist = metric
    if init is not None:
        Y[0] = init
    D = dist(X, Y[None,0])
    Y[0] = X[np.argmin(D)] # assign nearest point
    D = dist(X, Y[None,0])
    imax = np.argmax(D)
    mdist[0] = D[imax]
    
    # greedy iteration:
    for i in range(1,K):
        Y[i] = X[imax]
        tmp = dist(X, Y[None,i])
        D = np.minimum(D, tmp)
        imax = np.argmax(D)
        mdist[i] = D[imax]
    if out_dist:
        return Y, mdist
    else:
        return Y

def inv_hpd(M):
    
    """Computes the matrix inverse of a sequence of HPD matrices.

    Input is a (..,n,n) array M. Output is an array of the same shape as the
    input.
    """

    # sanity checks:
    assert isinstance(M, np.ndarray)
    assert M.ndim >= 2
    assert M.shape[-2] == M.shape[-1]
    assert M.dtype in [np.float64, np.complex128]

    shape = M.shape
    M = M.reshape((-1, *M.shape[-2:]))
    out = np.empty_like(M)
    if M.shape[-1] == 3:
        out = Descriptor_Cython.Geometric_Mean_Utils.inv_3x3(M, out).reshape(shape)
    else:
        out = Descriptor_Cython.Geometric_Mean_Utils.inv_hpd(M, out).reshape(shape)
    return out

def matrix_sym2vec(s):
    """
    Returns a matrix A such that |A*vec(M)|_2 = |M|_F for each 
    symmetric (s,s) matrix M.
    """
    assert isinstance(s, int) and s > 0
    res = np.zeros((s*(s+1)//2,s*s))
    sqrt2 = np.sqrt(0.5)
    ind = 0
    for i in range(s):
        res[ind,i*(s+1)] = 1.0
        for j in range(i+1,s):
            res[ind+j-i,i*s+j] = res[ind+j-i,j*s+i] = sqrt2
        ind += s-i
    return res

'Riemannian Mean Computation based on a gradient descent applied on equation (1) (Refer to Notebook)'


def Riemannian_Distance(A,B):
    return np.linalg.norm(logm(sqrtm(np.linalg.inv(A))@B@sqrtm(np.linalg.inv(A))),ord= 'fro')

' Vectorized matrix logarithm '

def Logm_vec(Matrix_Array):
    return np.array(list(map(logm,Matrix_Array)))

' Riemannian Mean Approximation via gradient descent with adaptive step size '

def Gradient_Descent_Mean(Matrix_Array,prec = 1e-3,Max_Iter = 100):
    'Intialization'
    tau = 1
    N = Matrix_Array.shape[0]
    starttime = time.time()
    X_0 = Matrix_Array[0]
    X_1 = X_0@expm(-tau*np.sum(Logm_vec(np.dot(np.linalg.inv(Matrix_Array),X_0)),axis = 0)/N)
    k = 0
    while Riemannian_Distance(X_1,X_0) > prec or k >= Max_Iter:
        X_0 = X_1.copy()
        X_1 = sqrtm(X_0)@expm(-tau*np.sum(Logm_vec(sqrtm(X_0)@np.dot(np.linalg.inv(Matrix_Array),sqrtm(X_0))),axis = 0)/N)@sqrtm(X_0)
        k += 1
        tau = 1/(k+1)
    t_end = time.time()-starttime
    print('Completed sucessfully')
    print('\n Required {:*^10} Iterations'.format(k))
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return X_1,t_end,k

' First dimension is the number of train data for Covariance Descriptor '   

def Return_Descriptors(f_cov,Region_Mask,Train_Sample,save = True,Region_num = 2,K = 5,iter = 200):
    Region_Descriptors= np.zeros((Region_num,300,3,K,7*7))
    starttime = time.time()
    for sample in range(Train_Sample):
        for chan in range(3):
            f_pd = f_cov[sample][:,:,chan,:,:]
            Mask = Region_Mask[sample][:,:]
            print('Extract Descriptors from Data_Sample {:*^10}'.format(sample))
            for k in range(Region_num):
                print('Extract Descriptors from Region {:*^10}'.format(k))
                Ft_Region = f_pd[Mask == k]
                ' Pick Random Cov Matrices '
                Region_Descriptors[k,sample,chan,:,:] = K_means(Ft_Region,K,Max_Iter=iter)
    Endtime = time.time()-starttime
    print('Finished in ---{:*^10}---'.format(Endtime))
    if save == True:
        np.save('/src/Covariance_Descriptor/Prototypes/Region_Descriptors_'+str(K),Region_Descriptors) 
    return Region_Descriptors

from scipy.linalg import schur


def SPD_Mean_Quadratic_1(Matrix_Array,Max_Iter = 120,tol=1e-4,Aut = True):
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    N,d = Matrix_Array.shape[0:2]
    
    'Initilizre Cholesky Decomposition'
    
    
    'Store Matrix L and its Inverse'
    
        
    List_Cholesky = np.zeros_like(Matrix_Array)  
    nuold = 100000
    
    'Initilize with arithmetic mean'
    
    Mean = np.zeros((d,d))
    for k in range(N):
        List_Cholesky[k] = cholesky(Matrix_Array[k])
    #    Mean             += Matrix_Array[k]/N 
    Mean = np.mean(Matrix_Array,axis = 0)
    print(List_Cholesky)
        #A1 = np.zeros_like(A.copy())
    #Mean = Cheap_Mean(Matrix_Array)
    starttime = time.time()
    for It in range(Max_Iter):
        
        R0  = cholesky(Mean)
        iR0 = np.linalg.inv(R0)
        
        'Initilize LIst for Cholesky '
        
        U_d = np.zeros_like(Matrix_Array)
        V_d = np.zeros((N,d))
        
        for k in range(N):
            
            Z        = List_Cholesky[k]@iR0
            V,U      = schur(Z.T@Z)
            U_d[k]     = U
            V_d[k]   = np.diag(V)
        
        if Aut == True:
            beta  = 0
            gamma = 0
            for s in range(N):
                ch = np.max(V_d[s])/np.min(V_d[s])
                if np.abs(ch-1) < 0.5:
                    dh = np.log1p(ch-1)/(ch-1)
                else:
                    dh = np.log(ch)/(ch-1)
                beta  += dh
                gamma += ch*dh
            theta = 2/(gamma + beta)
        S = np.zeros((d,d))
        for k in range(N):
            T = U_d[k]@np.diag(np.log(V_d[k]))@U_d[k].T
            S += (T+(T.T))/2
        Vs,Us = schur(S)
        Z     = np.diag(np.exp(np.diag(Vs*theta/2)))@(Us.T)@R0
        Mean  = Z.T@Z
        
        'Compute norm of S'
        
        nu    = np.max(np.abs(np.diag(Vs)))
        print(nu)
        
        if nu < np.linalg.norm(Mean)*tol or nu > nuold:
            print('\n Required {:*^10} Iterations'.format(It))
            break
        nuold = nu
    t_end = time.time()-starttime
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,It,t_end



