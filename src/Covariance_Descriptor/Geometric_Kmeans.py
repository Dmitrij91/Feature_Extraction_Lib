
from sklearn.cluster import k_means as kmeans
from Cython.cython_tools import distancematrix_stein as dis_stein
import linalgh
from tools import matrix_sym2vec
from Cython import cython_distance as cdistance
import numpy as np
import time
from Cython.Geometric_Mean_Utils import Bini_Riemann,Riemann_Update_Cluster,distance_Riemmann,sqrtm,logm


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
        # Cumpute Clusters 
        # Initialize G by log-Euclidean kmeans:
        # Transfer the Frobenius inner product to Euclidian
        A = matrix_sym2vec(s)
        # Matrix_Log,such that |X|_2 = |log f_pd|_F, == vec(log_f_pd)
        X = linalgh.logmh(f_pd).reshape((N_Pixel,s*s)).dot(A.T)
        # K_means on Eucliadian space == Tangential space log f_pd
        X = kmeans(X, K, n_init=n_init_kmeans)[0]
        # Transform to Manifold by exp such that G = exp(log X_1),...exp(log_X_K)
        A = A.T.dot(linalgh.invh(A.dot(A.T)))
        Clusters = linalgh.expmh(np.einsum('kl,jl->jk', A, X).reshape((K,s,s))) 
        if Metric == 'Log_Det':
            for it in range(Max_Iter):
                D = dis_stein(f_pd, Clusters)
                W[range(N_Pixel),np.argmin(D, axis=-1)] = 1.0 # assignment by NN
                if np.allclose(W_old, W):
                    break
                for j in range(Max_Iter):
                    mcG = cdistance.riccatiG(W/np.sum(W, axis=0), f_pd, Clusters)
                    grad = Clusters[0] - Clusters[0].dot(mcG[0]).dot(Clusters[0])
                    grad = np.sqrt(np.mean(np.abs(grad) ** 2))
                    Clusters = inv_hpd(mcG)
                    if grad < tol_mean:
                        break
                W_old = W
        elif Metric == 'Riemmann':
            for it in range(Max_Iter):
                D = distance_Riemmann(f_pd, Clusters)
                W[range(N_Pixel),np.argmin(D, axis=-1)] = 1.0 # assignment by NN
                if np.allclose(W_old, W):
                    break
                Clusters = Riemann_Update_Cluster(W/np.sum(W, axis=0), f_pd, Clusters)
                W_old = W
    return Clusters.reshape(-1,s*s)

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
        out = cla.inv_3x3(M, out).reshape(shape)
    else:
        out = inv_hpd(M, out).reshape(shape)
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

'Riemannian Mean Computation based on a gradient descent applied on equation (1)'

def Logm_vec(Matrix_Array):
    return np.array(list(map(logm,Matrix_Array)))

def Riemannian_Distance(A,B):
    return np.linalg.norm(logm(sqrtm(np.linalg.inv(A))@B@sqrtm(np.linalg.inv(A))),ord= 'fro')

def Gradient_Descent_Mean(Matrix_Array,prec = 20,Max_Iter = 1000):
    'Intialization'
    tau = 1
    X_0 = Matrix_Array[0]
    starttime = time.time()
    X_1 = X_0@expm(-tau*np.sum(Logm_vec(np.dot(np.linalg.inv(Matrix_Array),X_0)),axis = 0))
    k = 0
    while np.linalg.norm(X_1-X_0)/np.linalg.norm(X_0) > prec or k < Max_Iter:
        X_0 = X_1.copy()
        X_1 = sqrtm(X_0)@expm(-tau*np.sum(Logm_vec(sqrtm(X_0)@np.dot(np.linalg.inv(Matrix_Array),sqrtm(X_0))),axis = 0))@sqrtm(X_0)
        k += 1
        tau = 1/(k+1)
    t_end = time.time()-starttime
    print('Completed sucessfully')
    print('\n Required {:*^10} Iterations'.format(k))
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return X_1,t_end,k
    

def Return_Descriptors(f_cov,layer_mask,Scan,save = True,Layers = 14):
    assert f_cov.shape[0] == layer_mask.shape[0]
    assert f_cov.shape[1] == layer_mask.shape[1]
    assert f_cov.shape[2] == layer_mask.shape[2]
    Layer_Descriptors= np.zeros((Layers,10,20,10*10))
    Features = np.zeros((Layers,))
    starttime = time.time()
    for i,scan in enumerate(Scan):
        f_pd = f_cov[10:cropZ[1],5:layer_mask.shape[1]-50,scan-2:scan+2,:,:]
        Mask = layer_mask[10:cropZ[1],5:layer_mask.shape[1]-50,scan-2:scan+2]
        
        ' Reduce Size randomly along X axis and each Scan '
            
        Index_x = np.random.choice(Mask.shape[1],np.int(Mask.shape[1]/8))
        
        f_pd = f_pd[:,Index_x,:,:,:]
        Mask = Mask[:,Index_x,:]
        print('Extract Descriptors from scan {:*^10}'.format(scan))
        for k in range(Layers):
            print('Extract Descriptors from Layer {:*^10}'.format(k))
            
            Ft_layer = f_pd[Mask == k]
            print(Ft_layer.shape)
            ' Pick Random Cov Matrices '
    #for f_pd_noise,l in zip(List,Scan):
    #    for k in range(14):
    #        Mask = layer_mask[0:cropZ[1],:,l]
    #        print(l,s)
    #        Ft_layer = f_pd[0:cropZ[1],:,:,:][Mask == k]
            print(Ft_layer.shape)
            Layer_Descriptors[k,i,:,:] = K_means_python(Ft_layer,20,200)
    Endtime = time.time()-starttime
    print('Finished in ---{:*^10}---'.format(Endtime))
    if save == True:
        np.save('Layer_Descriptors_Stein_new_P5_13',Layer_Descriptors) 
    return Layer_Descriptors


