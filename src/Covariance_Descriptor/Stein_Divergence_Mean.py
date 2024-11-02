from Geometric_Kmeans import matrix_sym2vec
import Descriptor_Cython
import numpy as np
import time
from scipy.linalg import expm,logm
from Cheap_Mean import Cheap_Mean


def divergence_stein(X,Y):
    """Subroutine for computing the Stein divergence of two SPD matrices X and Y."""
    return np.log(LA.det(0.5*(X+Y)))-0.5*np.log(LA.det(X.dot(Y)))
    
def Stein_Mean(Matrix_Array,Max_Iter = 5000,tol=1e-10,Init = 'Arithmetic'):
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    N,d = Matrix_Array.shape[0:2]
    
    'Initialization'
    if Init == 'Arithmetic':
        Mean = np.zeros((d,d))
        Mean += np.sum(Matrix_Array,axis = 0 )/N 
    elif Init == 'Cheap_Mean':
        Mean = Cheap_Mean(Matrix_Array)
    tau = 1
    starttime = time.time()
    for k in range(Max_Iter):
        tau = 1/(k+1)
        R = np.zeros((d,d))
        for s in range(N):
            R += np.linalg.inv((Mean+Matrix_Array[s])/2)/N
        Mean_old = Mean.copy()
        Mean     = np.linalg.inv(R)
        if np.linalg.norm((Mean - Mean_old))/np.linalg.norm(Mean_old) < tol:
            print('\n Required {:*^10} Iterations'.format(k))
            break
    t_end = time.time()-starttime
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,k,t_end
    
def Stein_Mean_Geom(Matrix_Array,Max_Iter = 5000,tol=1e-10,Init = 'Arithmetic'):
    assert Matrix_Array.shape[1] == Matrix_Array.shape[2]
    N,d = Matrix_Array.shape[0:2]

    'Initialization'
    if Init == 'Arithmetic':
        Mean = np.zeros((d,d))
        Mean += np.sum(Matrix_Array,axis = 0 )/N 
    elif Init == 'Cheap_Mean':
        Mean = Cheap_Mean(Matrix_Array)
    #Mean = Cheap_Mean(Matrix_Array)
    tau = 1
    starttime = time.time()
    for k in range(Max_Iter):
        tau = 1/(k+1)
        'Compute Gradient'
        R = np.zeros((d,d))
        for s in range(N):
            R += np.linalg.inv((Mean+Matrix_Array[s])/2)/N
        Mean_old = Mean.copy()
        Mean     = Mean_old-tau* (1/2)*(Mean_old@R@Mean_old-Mean_old)
        if np.linalg.norm((Mean - Mean_old))/np.linalg.norm(Mean_old) < tol:
            print('\n Required {:*^10} Iterations'.format(k))
            break
    t_end = time.time()-starttime
    print('\n')
    print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
    return Mean,k,t_end

def Log_Euclid_Mean(Array_Mat):
    return expm(np.asarray(list(map(logm,Array_Mat))).mean(axis=0))

def Log_Euclid_Distance(Matrix_Features,Prototypes):
    
    ''' Tranform Matrix Array to upper triangular part of vector arrays
       by rescaling with factor 1/sqrt(2) see Arsigny'''
    
    x,y,z,s = Matrix_Features.shape[0:4]
    
    ' Rescaling Matrix '
    
    A = matrix_sym2vec(s)
    
    ' Rescaled Vector Array '
    
    Prototypes_Vec   =  np.zeros((Prototypes.shape[0],Prototypes.shape[1],int((s+1)*s/2)))
    
    for k in range(Prototypes.shape[0]):
        Prototypes_Vec[k,:,:] = Descriptor_Cython.Geometric_Mean_Utils.logmh(Prototypes.astype(np.double)[k,:,:,:]).reshape(Prototypes.shape[1],s*s).dot(A.T)
    
    Matrix_Array_Vec = Descriptor_Cython.Geometric_Mean_Utils.logmh(Matrix_Features.astype(np.double).reshape((x*y*z,s,s))).reshape(x*y*z,s*s).dot(A.T)
    
    'Inititlize Distance Matrix'
    
    Distance = np.zeros((x*y*z,Prototypes.shape[0],Prototypes.shape[1]))
    
    for k in range(Prototypes.shape[1]):
        Distance[:,:,k] = Descriptor_Cython.Geometric_Mean_Utils.cdistance(Matrix_Array_Vec,Prototypes_Vec[:,k,:])
    return Distance.reshape(x,y,z,Prototypes.shape[0],Prototypes.shape[1])
    
            
    
 