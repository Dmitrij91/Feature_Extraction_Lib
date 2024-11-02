' Returnes Cheap Mean Initializatuion by (3), see Jupyter Notebook '
import numpy as np
from scipy.linalg import expm,logm,inv,sqrtm


# Todo, make computation in parallel

def Cheap_Mean(Matrix_Array,max_iter = 10):
    Cheap_Mean_Init = np.zeros(Matrix_Array.shape)
    for k in range(max_iter):
        Cheap_Mean_Init = Fix_Iter_Cheap(Matrix_Array)
    return Cheap_Mean_Init

def Fix_Iter_Cheap(Matrix_Array):
    out = np.zeros(Matrix_Array.shape)
    for k in range(Matrix_Array.shape[0]):
        aux_1 = np.real(sqrtm(Matrix_Array[k,:,:]))
        aux_2 = np.einsum('ij,kjs -> kis',inv(aux_1),Matrix_Array)
        aux_2 = np.einsum('kij,js -> kis',aux_2,inv(aux_1))
        out[k,:,:] = aux_1@expm(np.mean(np.array(list(map(logm,aux_2))),axis = 0))@aux_1
    return out