from sklearn.cluster import k_means as kmeans
from Cython.cython_tools import distancematrix_stein as dis_stein
import linalgh
from tools import matrix_sym2vec
from Cython import cython_distance as cdistance
import numpy as np
import time
from Cython.Geometric_Mean_Utils import Bini_Riemann,Riemann_Update_Cluster,distance_Riemmann,sqrtm,logm

def eval_descriptors(layer_descriptors, f_pd):
    """
    Computes data terms, given a list of layer descriptors. For each location, take the min over all 
    descriptors.
    """
    start = time.perf_counter()
    Num = layer_descriptors.shape[1]
    D = []
    for k in range(0,15):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_00_Distance_Stein',D_12)
    del D_12
    D = []
    for k in range(15,30):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_15_Distance_Stein',D_12)
    del D_12
    D = []
    for k in range(30,45):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_30_Distance_Stein',D_12)
    del D_12
    for k in range(45,60):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_45_Distance_Stein',D_12)
    del D_12
    D = []
    for k in range(60,75):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_60_Distance_Stein',D_12)
    del D_12
    D_12 = np.array(D).min(0)
    D = []
    for k in range(75,90):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_75_Distance_Stein',D_12)
    del D_12
    D = []
    for k in range(90,105):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_11 = np.array(D).min(0)
    np.save('D_90_Distance_Stein',D_11)
    del D_11
    D = []
    for k in range(105,120):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_120_Distance',D_12)
    del D_12
    D = []
    for k in range(120,135):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_13 = np.array(D).min(0)
    np.save('D_135_Distance',D_13)
    del D_13
    D = []
    for k in range(135,150):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_11 = np.array(D).min(0)
    np.save('D_150_Distance',D_11)
    del D_11
    D = []
    for k in range(150,165):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_12 = np.array(D).min(0)
    np.save('D_165_Distance',D_12)
    del D_12
    D = []
    for k in range(165,180):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_13 = np.array(D).min(0)
    np.save('D_180_Distance',D_13)
    del D_13
    D = []
    for k in range(180,190):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_5 = np.array(D).min(0)
    np.save('D_190_Distance',D_5)
    del D_5
    D = []
    for k in range(190,200):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(distance_Riemmann(f_pd, layer_descriptors[:,k,:,:]))
    D_6 = np.array(D).min(0)
    np.save('D_200_Distance',D_6)
    del D_6
    #D_Total = np.array([D_1,D_2,D_3,D_4,D_5,D_6]).min(0)
    #Num_1 = layer_descriptors_1.shape[1]
    #layers = np.zeros((14,120,9,9))
    #for k in range(Num_1):
    #    layers[3:7,:,:,:] = layer_descriptors_1
    #    D_1.append(tools.distance_matrix(f_pd[:,:], layers[:,k,:,:], metric='Stein', angles=None))
    #D_1 = np.array(D_1).min(0)
    #print("Computed distance matrix in {:.2f} s".format(time.perf_counter()-start))

    return 0
    
def eval_descriptors_Stein(layer_descriptors, f_pd):
    """
    Computes data terms, given a list of layer descriptors. For each location, take the min over all 
    descriptors.
    """
    start = time.perf_counter()
    Num = layer_descriptors.shape[1]
    D = []
    k = 0
    for k in range(0,15):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_00_Stein',D_12)
    del D_12
    D = []
    for k in range(15,30):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_15_Stein',D_12)
    del D_12
    D = []
    for k in range(30,45):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_30_Stein',D_12)
    del D_12
    for k in range(45,60):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_45_Stein',D_12)
    del D_12
    D = []
    for k in range(60,75):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_60_Stein',D_12)
    del D_12
    D_12 = np.array(D).min(0)
    D = []
    for k in range(75,90):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_75_Stein',D_12)
    del D_12
    D = []
    for k in range(90,100):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_11 = np.array(D).min(0)
    np.save('Descriptors_Stein_OCT/D_90_Stein',D_11)
    del D_11
    D = []
    for k in range(105,120):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('D_120_Stein',D_12)
    del D_12
    D = []
    for k in range(120,135):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_13 = np.array(D).min(0)
    np.save('D_135_Stein',D_13)
    del D_13
    D = []
    for k in range(135,150):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_11 = np.array(D).min(0)
    np.save('D_150_Stein',D_11)
    del D_11
    D = []
    for k in range(150,165):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_12 = np.array(D).min(0)
    np.save('D_165_Stein',D_12)
    del D_12
    D = []
    for k in range(165,180):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_13 = np.array(D).min(0)
    np.save('D_180_Stein',D_13)
    del D_13
    D = []
    for k in range(180,190):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_5 = np.array(D).min(0)
    np.save('D_190_Stein',D_5)
    del D_5
    D = []
    for k in range(190,200):
        print("Computed distance matrix after layers {:.2f} s".format(k))
        D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,k,:,:], metric='Stein', angles=None))
    D_6 = np.array(D).min(0)
    np.save('D_200_Stein',D_6)
    del D_6
    #D_Total = np.array([D_1,D_2,D_3,D_4,D_5,D_6]).min(0)
    #Num_1 = layer_descriptors_1.shape[1]
    #layers = np.zeros((14,120,9,9))
    #for k in range(Num_1):
    #    layers[3:7,:,:,:] = layer_descriptors_1
    #    D_1.append(tools.distance_matrix(f_pd[:,:], layers[:,k,:,:], metric='Stein', angles=None))
    #D_1 = np.array(D_1).min(0)
    #print("Computed distance matrix in {:.2f} s".format(time.perf_counter()-start))

    return 0
    
def Return_Time_Means(f_pd, K = 1,Max_Iter = 400,Method = 'Log_Euclid'):
    s = f_pd.shape[2]
    N_Pixel = f_pd.shape[0]
    Random_Indices = np.random.choice(N_Pixel,K)
    # Initialization of K random Centroids
    Clusters = f_pd[Random_Indices]
    tol_mean = 1e-4
    Start = time.time()
    if Method == 'Log_Euclid': 
        # Cumpute Clusters 
        # Initialize G by log-Euclidean kmeans:
        # Transfer the Frobenius inner product to Euclidian
        A = matrix_sym2vec(s)
        #print(A)
        #print(f_pd)
        # Matrix_Log,such that |X|_2 = |log f_pd|_F, == vec(log_f_pd)
        X = linalgh.logmh(f_pd).reshape((N_Pixel,s*s)).dot(A.T)
        #print(X,'X') 
        # K_means on Eucliadian space == Tangential space log f_pd
        X = kmeans(X, K, n_init=n_init_kmeans)[0]
        # Transform to Manifold by exp such that G = exp(log X_1),...exp(log_X_K)
        A = A.T.dot(linalgh.invh(A.dot(A.T)))
        Start = time.time()
        Clusters = linalgh.expmh(np.einsum('kl,jl->jk', A, X).reshape((K,s,s)))
        time_Log_Euclid = time.time()-Start
        Clusters_Log_Eucl = Clusters.copy()
   # elif init == 'random':
   #         G = linalgh.expmh(np.random.randn(n,s,s))
   # elif init == 'random2':
   #         %%G = np.random.rand(n,s,s)
   #         G = np.einsum('lij,kj->lik', G, G)   
        
        W = np.ones((N_Pixel,1))
        Start_S = time.time()
        for j in range(Max_Iter):
            mcG = cdistance.riccatiG(W/np.sum(W, axis=0), f_pd, Clusters)
            grad = Clusters[0] - Clusters[0].dot(mcG[0]).dot(Clusters[0])
            grad = np.sqrt(np.mean(np.abs(grad) ** 2))
            Clusters_Stein = inv_hpd(mcG)
            Clusters = Clusters_Stein.copy()
            if grad < tol_mean:
                time_Stein = time.time()-Start_S
                break
                
        Clusters_Riemann,time_Riemann = Bini_Riemann(f_pd, Clusters_Log_Eucl[0,:,:])
    return Clusters_Stein.reshape(-1,s,s),Clusters_Riemann.reshape(-1,s,s),Clusters_Log_Eucl.reshape(-1,s,s),time_Log_Euclid,time_Riemann,time_Stein
    
    
def eval_descriptors_Stein(layer_descriptors, f_pd,count):
    """
    Computes data terms, given a list of layer descriptors. For each location, take the min over all 
    descriptors.
    """
    start = time.perf_counter()
    Num = layer_descriptors.shape[1]
    D = []
    l = 0
    print("Computed distance matrix after layers {:.2f} s".format(l))
    if Metric == 'Log_Det':
        while l < count:
            D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,l,:,:], metric='Stein', angles=None))
            if l%10 == 9 and l!= 0:
                print("Current State {:.2f}".format(l))
                D_12 = np.array(D).min(0)
                #np.save('Descriptors_Stein_OCT/D_Stein'+str(l),D_12)
                #del D_12
                D = []
            l = l+1
    elif Metric == 'Riemann':
        while l < count:
            D.append(distance_Riemmann(f_pd, layer_descriptors[:,l,:,:]))
            if l%10 == 9 and l!= 0:
                print("Current State {:.2f}".format(l))
                D_12 = np.array(D).min(0)
                #np.save('Descriptors_Stein_OCT/D_Stein'+str(l),D_12)
                #del D_12
                D = []
            l = l+1
    elif Metric == 'Riemann':
            D.append(tools.distance_matrix(f_pd[:,:], layer_descriptors[:,l,:,:], metric='Stein', angles=None))
            if l%10 == 9 and l!= 0:
                print("Current State {:.2f}".format(l))
                D_12 = np.array(D).min(0)
                #np.save('Descriptors_Stein_OCT/D_Stein'+str(l),D_12)
                #del D_12
                D = []
            l = l+1
