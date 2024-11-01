from Cython import cython_distance as cdistance
from Geometric_Kmeans import  inv_hpd 
import scipy.linalg as LA

def geometric_mean(M, th_JAD=1e-6, th=1e-10):
    """Computation of the geometric mean of SPD matrices using the Joint Approximate Diagonalization.
    
    Parameters
    ----------
    M : array
        A (n,s,s) array, where M[i,:,:] respectively is a s-by-s SPD matrix.
    th_JAD : float
        Threshold for computing the Joint Approximate Diagonalization.
    th : float
        Threshold.
    
    Returns
    -------
    mean_M : array
        A (s,s) array representing the geometric mean of the n matrices stored in M.
    """
    
    # sanity checks:
    M = np.asarray(M)
    assert M.ndim == 3
    assert M.shape[1] == M.shape[2]
    if __debug__:
        for i in range(M.shape[0]):
            assert np.allclose(M[i],M[i].T), "M[i] is not symmetric!"
            assert np.all(LA.eigvalsh(M[i]) > 0.0), "M[i] is not pd!"
    assert th_JAD > 0.0, "th_JAD should be positive!"
    assert th > 0.0, "th should be positive!"
    
    n = M.shape[0]
    starttime = time.time()
    B = compute_JAD(M, th_JAD)
    L = np.empty_like(M)
    for i in range(n):
        L[i] = logm(B.dot(M[i]).dot(B.T))
    D = expm(np.mean(L, axis=0))
    d = np.diag(D)
    It = 0
    while LA.norm(d-1, np.inf) > th:
        B = np.diag(d**(-0.5)).dot(B)
        for i in range(n):
            L[i] = logm(B.dot(M[i]).dot(B.T))
        D = expm(np.mean(L, axis=0))
        d = np.diag(D)
        It += 1
    A = LA.inv(B)
    Mean = A.dot(D).dot(A.T)
    
    t_end = time.time()-starttime
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,It,t_end
    
def compute_JAD(M, th):
    """Subroutine for computing the Joint Approximate Diagonalization."""
    n = M.shape[0]
    s = M.shape[1]
    B = np.zeros((s,s))
    for i in range(n):
        B += logm(M[i])
    B = expm(B/n)
    resmax = th + 1.0
    while resmax > th:
        resmax = 0.0
        for i in range(s):
            for j in range(i+1,s):
                Bij = B[[i,j]] # (2,s) submatrix of B with rows i and j
                # P = 1/n * sum_r (Bij * M[r] * Bij^T) / (B * M[r] * B.T)_ii
                P = np.mean(np.einsum('ril,jl->rij', np.einsum('ik,rkl->ril', Bij, M), Bij) /
                            np.einsum('rl,l->r', np.einsum('k,rkl->rl', B[i], M), B[i])[:,None,None],
                            axis=0)
                # Q = 1/n * sum_r (Bij * M[r] * Bij^T) / (B * M[r] * B.T)_jj
                Q = np.mean(np.einsum('ril,jl->rij', np.einsum('ik,rkl->ril', Bij, M), Bij) /
                            np.einsum('rl,l->r', np.einsum('k,rkl->rl', B[j], M), B[j])[:,None,None],
                            axis=0)
                T = eigh2(P,Q) # generalized eigenvectors of (P,Q)
                TP = T.T.dot(P).dot(T)
                TQ = T.T.dot(Q).dot(T)
                if TP[1,1]*TQ[0,0] < TP[0,0]*TQ[1,1]:
                    T = T[:,::-1]
                B[[i,j]] = T.T.dot(Bij)
                resmax = max(resmax, np.abs(P[0,1]))       
    return B
    
def eigh2(A,B):
    """Generalized eigenvectors of (A,B) for 2x2 matrices."""
    a = B[0,0] * B[1,1] - B[0,1] * B[0,1]
    b = A[0,0] * B[1,1] + A[1,1] * B[0,0] - 2.0 * A[0,1] * B[0,1]
    c = A[0,0] * A[1,1] - A[0,1] * A[0,1]
    sqrt_discriminant = np.sqrt(b*b-4.0*a*c)
    l1 = 0.5 * (b - sqrt_discriminant) / a
    l2 = 0.5 * (b + sqrt_discriminant) / a
    V = np.empty((2,2))
    if sqrt_discriminant == 0.0: # maybe replace by: sqrt_discriminant/a < 1e-14
        # This is the very unlikely but ugly case where A = alpha * B
        V[0,0] = 1.0 / np.sqrt(B[0,0])
        V[1,0] = 0.0
        V[0,1] = -B[0,1] / np.sqrt(B[0,0] * a)
        V[1,1] = np.sqrt(B[0,0] / a)
    else:
        if np.abs(l1*B[0,0]-A[0,0]) > np.abs(l1*B[1,1]-A[1,1]):
            V[0,0] = A[0,1] - l1 * B[0,1]
            V[1,0] = l1 * B[0,0] - A[0,0]
        else:
            V[0,0] = l1 * B[1,1] - A[1,1]
            V[1,0] = A[0,1] - l1 * B[0,1]
        V[:,0] /= np.sqrt(V[:,0].dot(B.dot(V[:,0])))
        if np.abs(l2*B[0,0]-A[0,0]) > np.abs(l2*B[1,1]-A[1,1]):
            V[0,1] = A[0,1] - l2 * B[0,1]
            V[1,1] = l2 * B[0,0] - A[0,0]
        else:
            V[0,1] = l2 * B[1,1] - A[1,1]
            V[1,1] = A[0,1] - l2 * B[0,1]
        V[:,1] /= np.sqrt(V[:,1].dot(B.dot(V[:,1])))
    return V
    
def divergence_stein(X,Y):
    """Subroutine for computing the Stein divergence of two SPD matrices X and Y."""
    return np.log(LA.det(0.5*(X+Y)))-0.5*np.log(LA.det(X.dot(Y)))
    
def Stein_Mean(Matrix_Array,Max_Iter = 5000,tol=1e-10):
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    N,d = Matrix_Array.shape[0:2]
    
    'Initialization'
    
    Mean = np.zeros((d,d))
    Mean += np.sum(Matrix_Array,axis = 0 )/N 
    Mean = Cheap_Mean(Matrix_Array)
    tau = 1
    starttime = time.time()
    for k in range(Max_Iter):
        #tau = 1/(k+1)
        'Compute Gradient'
        Grad = np.zeros((d,d))
        R = np.zeros((d,d))
        for s in range(N):
            R += np.linalg.inv((Mean+Matrix_Array[s])/2)/N
        Mean_old = Mean.copy()
        #Mean     = Mean_old-tau* (1/2)*(Mean_old@R@Mean_old-Mean_old)
        Mean     = np.linalg.inv(R)
        if np.linalg.norm((Mean - Mean_old))/np.linalg.norm(Mean_old) < tol:
            print('\n Required {:*^10} Iterations'.format(k))
            break
    t_end = time.time()-starttime
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,k,t_end
    
def Stein_Mean_Geom(Matrix_Array,Max_Iter = 5000,tol=1e-10):
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    N,d = Matrix_Array.shape[0:2]
    
    'Initialization'
    
    Mean = np.zeros((d,d))
    Mean += np.sum(Matrix_Array,axis = 0 )/N 
    #Mean = Cheap_Mean(Matrix_Array)
    tau = 1
    starttime = time.time()
    for k in range(Max_Iter):
        #tau = 1/(k+1)
        'Compute Gradient'
        Grad = np.zeros((d,d))
        R = np.zeros((d,d))
        for s in range(N):
            R += np.linalg.inv((Mean+Matrix_Array[s])/2)/N
        Mean_old = Mean.copy()
        #Mean     = Mean_old@expm((-tau* (1/2)*(Mean_old@R@Mean_old-Mean_old)))
        Mean     = Mean_old-tau* (1/2)*(Mean_old@R@Mean_old-Mean_old)
        if np.linalg.norm((Mean - Mean_old))/np.linalg.norm(Mean_old) < tol:
            print('\n Required {:*^10} Iterations'.format(k))
            break
    t_end = time.time()-starttime
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,k,t_end
    
def Log_Euclid_Distance(Matrix_Array,Prototypes):
    
    ' Tranform Matrix Array to upper triangular part of vector arrays\
       by rescaling with factor 1/sqrt(2) see Arsigny'
    
    x,y,z,s = Matrix_Array.shape[0:4]
    
    ' Rescaling Matrix '
    
    A = matrix_sym2vec(s)
    
    
    ' Rescaled Vector Array '
    
    Prototypes_Vec   =  np.zeros((Prototypes.shape[0],Prototypes.shape[1],int((s+1)*s/2)))
    
    print(Prototypes_Vec.shape)
    
    for k in range(Prototypes.shape[0]):
        print(Prototypes_Vec.shape)
        Prototypes_Vec[k,:,:] = linalgh.logmh(Prototypes.astype(np.double)[k,:,:,:]).reshape(Prototypes.shape[1],s*s).dot(A.T)
    
    Matrix_Array_Vec = linalgh.logmh(f_pd.astype(np.double).reshape((x*y*z,s,s))).reshape(x*y*z,s*s).dot(A.T)
    
    'Inititlize Distance Matrix'
    
    Distance = np.zeros((x*y*z,Prototypes.shape[0],Prototypes.shape[1]))
    
    for k in range(Prototypes.shape[1]):
        Distance[:,:,k] = cdist(Matrix_Array_Vec,Prototypes_Vec[:,k,:])
    return Distance.reshape(x,y,z,Prototypes.shape[0],Prototypes.shape[1])
    
            
    
 
    

