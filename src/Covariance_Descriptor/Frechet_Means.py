import time
import numpy as np
from scipy.linalg import expm,logm,cholesky,schur
from Cython.Frechet_Means import sqrtm


class Matrix_Mean:
    def __init__(self,data):
        self.data = data
    def Log_Euclid(self):
        ' Quadratic Matrices Check ' 
        assert self.data.shape[1] == self.data.shape[2]
        N,d = self.data.shape[0:2]
        X = np.zeros((d,d))
        for k in range(N):
            try:
                cholesky(self.data[k,:,:])
                X += logm(self.data[k,:,:])
            except:
                print('Matrix not positive-definite!')
            starttime = time.time()
            X += logm(self.data[k,:,:])
            X /= N
        X = expm(X)
        t_end = time.time()-starttime
        print('\n')
        print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
        return X

    def Logm_vec(self):
        return np.array(list(map(logm,self.data)))

    def Riemannian_Distance(A,B):
        return np.linalg.norm(logm(sqrtm(np.linalg.inv(A))@B@sqrtm(np.linalg.inv(A))),ord= 'fro')

    def Gradient_Descent_Mean(self,prec = 20,Max_Iter = 1000):
        'Intialization'
        tau = 1
        X_0 = self.data[0]
        starttime = time.time()
        X_1 = X_0@expm(-tau*np.sum(Logm_vec(np.dot(np.linalg.inv(self.data),X_0)),axis = 0))
        k = 0
        while np.linalg.norm(X_1-X_0)/np.linalg.norm(X_0) > prec or k < Max_Iter:
            X_0 = X_1.copy()
            X_1 = sqrtm(X_0)@expm(-tau*np.sum(Logm_vec(sqrtm(X_0)@np.dot(np.linalg.inv(self.data),sqrtm(X_0))),axis = 0))@sqrtm(X_0)
            k += 1
            tau = 1/(k+1)
        t_end = time.time()-starttime
        print('Completed sucessfully')
        print('\n Required {:*^10} Iterations'.format(k))
        print('\n')
        print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
        return X_1,t_end,k



    ' References: \
    [1] D.A. Bini and B. Iannazzo, "A note on computing matrix geometric \
    means", Adv. Comput. Math., 35-2/4 (2011), pp. 175-192.'

    def Cheap_Mean(self,Max_Iter = 50,tol=1e-10):
        assert self.data.shape[1] == self.data.shape[2]
        N,d = self.data.shape[0:2]
        
        'Initilizre Cholesky Decomposition'
        
        
        'Store Matrix L and its Inverse'
        
            
        A  = self.data.copy()
        A1 = np.zeros_like(A.copy())
        starttime = time.time()
        for It in range(Max_Iter):
            List_Cholesky = np.zeros_like(A)
            List_Cholesky_Inv = np.zeros_like(A)
            for k in range(N):
                L     = cholesky(A[k])
                L_inv = np.linalg.inv(L) 
                List_Cholesky[k]     = L
                List_Cholesky_Inv[k] = L_inv
            for k in range(N):
                List_Cholesky_Inv[k] = np.linalg.inv((List_Cholesky[k])) 
                S = np.zeros((d,d))
                for l in range(N):
                    if l != k:
                        Z   = List_Cholesky[l]@List_Cholesky_Inv[k]
                        V,U = schur(Z.T@Z)
                        T   = U@np.diag(np.log(np.diag(V)))@U.T
                        S += (T.transpose()+T)/2
                V,U = schur((1/N)*S)
                T   = sqrtm(np.diag(np.exp(np.diag(V))))@U.T@List_Cholesky[k]
                A1[k] = (T.T)@T
            if np.linalg.norm(A1[0]-A[0])/np.linalg.norm(A[0]) < tol:
                print('\n Required {:*^10} Iterations'.format(k))
                break
            A = A1
        
        'Compute Mean'
        
        X = A[0]
        for k in range(1,N):
            X += A[k]
        t_end = time.time()-starttime
        print('Requered Iterations -------{:*^10}------- seconds'.format(t_end))
        
        return X/N

    ' References \
    [1] D.A. Bini and B. Iannazzo, "Computing the Karcher mean of symmetric \
    positive definite matrices", Linear Algebra Appl., 438-4 (2013), '

    def SPD_Mean_Quadratic_1(self,Max_Iter = 120,tol=1e-4,Aut = True):
        assert self.data.shape[1] == self.data.shape[2]
        N,d = self.data.shape[0:2]
        
        'Initilizre Cholesky Decomposition'
        
        
        'Store Matrix L and its Inverse'
        
            
        List_Cholesky = np.zeros_like(self.data)  
        nuold = 100000
        
        'Initilize with arithmetic mean'
        
        Mean = np.zeros((d,d))
        for k in range(N):
            List_Cholesky[k] = cholesky(self.data[k])
        #    Mean             += self.data[k]/N 
        Mean = np.mean(self.data,axis = 0)
        print(List_Cholesky)
            #A1 = np.zeros_like(A.copy())
        #Mean = Cheap_Mean(self.data)
        starttime = time.time()
        for It in range(Max_Iter):
            
            R0  = cholesky(Mean)
            iR0 = np.linalg.inv(R0)
            
            'Initilize LIst for Cholesky '
            
            U_d = np.zeros_like(self.data)
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
            