import numpy as np
from scipy.ndimage.filters import convolve, convolve1d
from scipy.special import binom
from scipy.ndimage.filters import gaussian_filter as gaussian
from skimage.color import rgb2hsv
from skimage.util import view_as_windows

"""
This class computes the covariance region descriptors for the given data using a set of filters.

This method extracts feature vectors for each data point (pixel) based on a provided list of predefined filters and filter masks. For each pixel, it computes the covariance matrix of the feature vectors within a `pxp` neighborhood around the pixel. 

### Predefined Filters:
The following filters are available for feature extraction:
- **'x'**: x-coordinate of the pixels (normalized to [0,1])
- **'y'**: y-coordinate of the pixels (normalized to [0,1])
- **'I0'**: intensity values at pixels (without smoothing)
- **'I'**: smoothed intensity values using a Gaussian filter
- **'Ix'**: first derivative with respect to the x-coordinate using a binomial filter
- **'Iy'**: first derivative with respect to the y-coordinate using a binomial filter
- **'Ixx'**: second derivative using a binomial filter
- **'Ixy'**: second derivative using a binomial filter
- **'Iyy'**: second derivative using a binomial filter
- **'M'**: result of applying a custom filter mask provided in the `filter_mask` list
- **'|Ix|'**: absolute value of the filter 'Ix'
- **'|Iy|'**: absolute value of the filter 'Iy'
- **'|gradI|'**: norm of the intensity gradient
- **'|Ixx|'**: absolute value of the filter 'Ixx'
- **'|Ixy|'**: absolute value of the filter 'Ixy'
- **'|Iyy|'**: absolute value of the filter 'Iyy'
- **'|M|'**: absolute value of the filter 'M'

All filters, except for 'x' and 'y', are applied channel-wise. For example, for an RGB image, applying the filters 'x', 'y', H,S,V,'I', 'Ix', 'Iy' results in feature vectors of the form: `x, y, R, G, B, Rx, Gx, Bx, Ry, Gy, By`. The derivatives are computed using a binomial filter with size `(p_filter, p_filter)`.

### Parameters:
- **self.data**: `array`
    - Input image data.
- **p_filter**: `int`
    - Size of the neighborhood for filters that use binomial filters for derivative calculations.
- **p_cov**: `int`
    - Size of the neighborhood used for covariance matrix computation.
- **filter_list**: `list of str`
    - List of predefined filters to be used for feature extraction.
- **eps_pd**: `float`
    - Regularization parameter. A value of `eps_pd * identity matrix` is added to each covariance matrix to ensure positive definiteness.
- **filter_masks**: `list of arrays`
    - Additional custom filters provided as filter masks.
- **subtract_mean**: `bool`
    - If `False`, the covariance matrix is computed without subtracting the mean from the feature vectors.

### Returns:
- **cov**: `array`
    - A 4D array `res[i, j, k, l]` where each element stores the `(k, l)` entry of the covariance matrix for the pixel at position `(i, j)`.
"""


class Features:
    def __init__(self,data,chanels_num,Scale_List = [7],subtract_mean = True,Hdim = False,filter_list = 'E,x,z,H,S,V,I,Ix,Iz,Ixx,Ixz,Izz,|gradI|',pad_x = 11,pad_y = 11,sigma_E = 1):
        self.data = data
        self.subtract_mean = subtract_mean
        self.chanels_num = chanels_num 
        self.filter_list = filter_list
        self.HDim = Hdim
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.sigma_E = sigma_E
        self.weights = filter_binomial(3,0,0)
        self.Scale_List = Scale_List # Max Response of filter outputs over scales
        self.f_vec = self.covariance_descriptor_3D(data,chanels_num,Scale_List,subtract_mean)

    ' Curvature Features using Weingarten map approximation '

    def Weingarten_Map(self,I,Mask_dim_x = 13,Mask_dim_y = 13):
        assert Mask_dim_x % 2 != 0 and Mask_dim_y % 2 != 0 
        
        N = (Mask_dim_x*Mask_dim_y)
        Mask = np.ones((Mask_dim_x,Mask_dim_y))/N

        Res_x = 1/I.shape[0]
        Res_y = 1/I.shape[1]


        Im_pred_vec = np.zeros((*I.shape,3))
        Im_pred_vec_mean = np.zeros((*I.shape,3))
        Im_pred_vec[:,:,2] = I[:,:]
        Im_pred_vec[:,:,0:2] = np.moveaxis(np.array(np.indices((Im_pred_vec.shape[0:2])),dtype = np.float32),0,-1)
        ' Add Resolution to the indices ' 
        Im_pred_vec[:,:,0:2] = np.einsum("ijk,k->ijk",Im_pred_vec[:,:,0:2],np.array([Res_x,Res_y]))
        Im_pred_vec_mean = convolve(Im_pred_vec,Mask[:,:,None])
        Im_pred_vec_mean_w = convolve(Im_pred_vec,Mask[:,:,None])

        Cov_Mat = np.zeros((*I.shape,3,3))

        Cov_Mat =  N*convolve(np.einsum('ijk,ijl->ijkl', Im_pred_vec, Im_pred_vec), Mask[:, :, None,None], mode="nearest")
        Cov_Mat -= np.einsum('ijk,ijl->ijkl', Im_pred_vec_mean_w, Im_pred_vec_mean)
        Cov_Mat -= np.einsum('ijl,ijk->ijkl', Im_pred_vec_mean_w, Im_pred_vec_mean)
        Cov_Mat += np.einsum('ijk,ijl->ijkl', Im_pred_vec_mean, Im_pred_vec_mean)

        N_space,T_space,_ = np.split(np.linalg.eigh(Cov_Mat)[1],(1,3),axis = 3)
        N_space = np.squeeze(N_space)



        " Weingarten_Map "

        W_Matrix = np.zeros((Cov_Mat.shape[0],Cov_Mat.shape[1],2,2))


        sigma = 3
        Y_shift = (Mask_dim_x+1)//2
        X_shift = (Mask_dim_y+1)//2

        # Pad Input and shift center window around each pixel
        X_diff = np.pad(Im_pred_vec,((X_shift-1,X_shift-1),(Y_shift-1,Y_shift-1),(0,0)))
        X_diff = view_as_windows(X_diff,(Mask_dim_x,Mask_dim_y,3),step=(1, 1, 1))
        X_diff = np.squeeze(X_diff)
        X_diff -= Im_pred_vec[:,:,None,None,:]


        N_space_pad = np.pad(N_space,((X_shift-1,X_shift-1),(Y_shift-1,Y_shift-1),(0,0)))
        N_space_pad = view_as_windows(N_space_pad,(Mask_dim_x,Mask_dim_y,3),step=(1, 1, 1))
        N_space_pad =  np.squeeze(N_space_pad)
        Delta_k = np.einsum('ijlk,ijsml->ijsmk',T_space[:,:,:,:],X_diff)
        Scalar_Mat = np.einsum('ijsmk,ijk->ijsm',N_space_pad,N_space)
        N_space_pad = N_space_pad*Scalar_Mat[:,:,:,:,None]-N_space[:,:,None,None,:]
        Theta_k = np.einsum('ijkl,ijsmk->ijsml',T_space,N_space_pad)
        Diag_W = np.exp(-np.einsum('ijkls,ijkls->ijkl',X_diff,X_diff)/(sigma**2))/(sigma**2)
        W_Matrix = -np.einsum('ijsmk,ijsml->ijkl',Theta_k*Diag_W[:,:,:,:,None],Delta_k)
        Theta_k = np.einsum('ijsmk,ijsml->ijkl',Delta_k*Diag_W[:,:,:,:,None],Delta_k)+0.1*np.eye(2)[None,None,:,:]
        Theta_k = np.linalg.inv(Theta_k.reshape(-1,2,2)).reshape(*Theta_k.shape)
        W_Matrix = np.einsum('ijsk,ijkm->ijsm',W_Matrix,Theta_k)
        W_Matrix = 0.5*(W_Matrix+np.swapaxes(W_Matrix,axis1=2,axis2=3))

        ' Compute Curvatures '
        nx,ny = W_Matrix.shape[0:2]
        mean_curvature = np.zeros((nx, ny))
        gaussian_curvature = np.zeros((nx, ny))
        principal_curvatures = np.zeros((nx, ny, 2))  # Store kappa1, kappa2

        # Compute eigenvalues and curvatures
        for i in range(nx):
            for j in range(ny):
                W = W_Matrix[i, j]
                # Compute eigenvalues (principal curvatures)
                kappa1, kappa2 = np.linalg.eigvalsh(W)  # Use eigvalsh for symmetric matrices
                principal_curvatures[i, j] = [kappa1, kappa2]
                # Mean and Gaussian curvatures
                mean_curvature[i, j] = 0.5*(kappa1 + kappa2)
                gaussian_curvature[i, j] = kappa1 * kappa2
            
        return [mean_curvature,gaussian_curvature,principal_curvatures[:, :, 0],principal_curvatures[:, :, 1]]
        

    ' Computes Entropy Features '

    def Entropy_Filter(self,I):
        I_window = np.exp(-(view_as_windows(np.pad(I,((int((self.pad_x-1)/2),\
            int((self.pad_x-1)/2)),(int((self.pad_y-1)/2),int((self.pad_y-1)/2)))),\
                       (self.pad_x,self.pad_y)).reshape(*I[:,:].shape,-1))/self.sigma_E)
        I_window = I_window/(np.sum(I_window,axis = 2)[:,:,None])
        return np.einsum('ijk,ijk->ij',I_window,-np.log(I_window))
      

    def covariance_from_fvec(self):
        """
        Subroutine for computing covariance matrices for given feature vectors.
        """
        if self.chanels_num == 1:
            if self.HDim == True:
                z,x,y,f = self.f_vec.shape
                res = np.empty((z,x,y,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    pz,px,py = self.weights.shape
                    weights_uniform = np.ones((pz,px,py)) / (px*py*pz)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for i in range(f):
                        fvec_mean[:,:,:,i] = convolve(self.f_vec[:,:,:,i], weights_uniform)
                        fvec_mean_w[:,:,:,i] = convolve(self.f_vec[:,:,:,i], self.weights) 
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,i,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], self.weights) \
                                - fvec_mean[:,:,:,i] * fvec_mean_w[:,:,:,j] \
                                - fvec_mean_w[:,:,:,i] * fvec_mean[:,:,:,j] \
                                + fvec_mean[:,:,:,i] * fvec_mean[:,:,:,j] * np.sum(self.weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,i,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], self.weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
            else:
                m,n,f = self.f_vec.shape
                res = np.empty((m,n,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    px,py = self.weights.shape
                    weights_uniform = np.ones((px,py)) / (px*py)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for i in range(f):
                        fvec_mean[:,:,i] = convolve(self.f_vec[:,:,i], weights_uniform)
                        fvec_mean_w[:,:,i] = convolve(self.f_vec[:,:,i], self.weights) 
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], self.weights) \
                                - fvec_mean[:,:,i] * fvec_mean_w[:,:,j] \
                                - fvec_mean_w[:,:,i] * fvec_mean[:,:,j] \
                                + fvec_mean[:,:,i] * fvec_mean[:,:,j] * np.sum(self.weights)
                            res[:,:,j,i] = res[:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], self.weights)
                            res[:,:,j,i] = res[:,:,i,j]
        else:
            if self.HDim == True:
                z,x,y,f = self.f_vec.shape
                res = np.empty((z,x,y,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    pz,px,py = self.weights.shape
                    weights_uniform = np.ones((pz,px,py)) / (px*py*pz)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for i in range(f):
                        fvec_mean[:,:,:,i] = convolve(self.f_vec[:,:,:,i], weights_uniform)
                        fvec_mean_w[:,:,:,i] = convolve(self.f_vec[:,:,:,i], self.weights) 
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:i,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], self.weights) \
                                - fvec_mean[:,:,:,i] * fvec_mean_w[:,:,:,j] \
                                - fvec_mean_w[:,:,:,i] * fvec_mean[:,:,:,j] \
                                + fvec_mean[:,:,:,i] * fvec_mean[:,:,:,j] * np.sum(self.weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], self.weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
            else:
                m,n,f = self.f_vec.shape
                res = np.empty((m,n,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    px,py = self.weights.shape
                    weights_uniform = np.ones((px,py)) / (px*py)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                
                    for i in range(f):
                        fvec_mean[:,:,i] = convolve(self.f_vec[:,:,i], weights_uniform)
                        fvec_mean_w[:,:,i] = convolve(self.f_vec[:,:,i], self.weights) 
                
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], self.weights) \
                                - fvec_mean[:,:,i] * fvec_mean_w[:,:,j] \
                                - fvec_mean_w[:,:,i] * fvec_mean[:,:,j] \
                                + fvec_mean[:,:,i] * fvec_mean[:,:,j] * np.sum(self.weights)
                            res[:,:,j,i] = res[:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], self.weights)
                            res[:,:,j,i] = res[:,:,i,j]
        return res

    def covariance_descriptor_3D(self,data,channel_num,Scale_List,subtract_mean,p_filter = 3, p_cov = 3,eps_pd = 0.0, filter_masks = [],HDim = False):
        

        mask = np.array([0.299,0.587,0.114])/(255.0) # Lift to Gray Values
        # sanity checks:
        assert isinstance(p_filter, int), "p should be an integer!"
        assert p_filter%2 == 1, "p_filter should be odd!"
        assert isinstance(p_cov, int), "p_cov should be odd!"
        assert p_cov%2 == 1, "p_cov should be odd!"
        assert eps_pd >= 0.0, "eps_pd should be non-negative!"
        assert isinstance(subtract_mean, bool) 
        assert isinstance(filter_masks, list)
        if __debug__:
            for mask in filter_masks:
                assert(mask.ndim == 2)
                assert(mask.shape[0]%2 == 1)
                assert(mask.shape[1]%2 == 1)
        assert isinstance(self.filter_list, (str,list))
        assert self.filter_list.count('M') == len(filter_masks)
        if __debug__:
            if isinstance(self.filter_list, list):
                for filt in self.filter_list:
                    assert isinstance(filt, str)
        
        if self.chanels_num == 1:
            if data.ndim == 2:
                data = data[:,:,None] # convert a (m,n) tensor to a (m,n,1) tensor
            if data.ndim == 3:
                data = data[:,:,:,None] # convert a (m,n,p) tensor to a (m,n,p,1) tensor
                
        if isinstance(self.filter_list, str):
            self.filter_list = self.filter_list.split(',')
        if 'x' in self.filter_list or 'y' in self.filter_list or 'z' in self.filter_list:
            ind = np.indices(data.shape[:-1])    

        # generate list of filter functions (which are applied afterwards):
        if HDim:
            filter_funcs = []
            ind_mask = 0
            for filt in self.filter_list:
                if filt == 'y':
                    filter_funcs.append([lambda I,res=ind[2][::-1]: ind[2]/I.shape[2]])
                elif filt == 'x':
                    filter_funcs.append([lambda I,res=ind[1][::-1]: ind[1][::-1]/I.shape[1]])
                elif filt == 'z':
                    filter_funcs.append([lambda I,res=ind[0][::-1]: ind[0][::-1]/I.shape[0]])
                elif filt == 'I0':
                    for j in range(data.shape[3]):
                        filter_funcs.append([lambda I,k=j: I[:,:,:,k] ])
                elif filt == 'I':
                    for j in range(data.shape[3]):
                        filter_funcs.append([
                            lambda I,k=j,h=filter_binomial_3D(p_filter,0,0,0): 
                                convolve(I[:,:,:,k],h)] )
                elif filt == 'Ix':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 1),Smooth,axis = 0)\
                                            ,Smooth,axis = 2))
                        filter_funcs.append(List)
                elif filt == 'Iy':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 2),Smooth,axis = 0)\
                                            ,Smooth,axis = 1))
                        filter_funcs.append(List)
                elif filt == 'Iz':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 0),Smooth,axis = 1)\
                                            ,Smooth,axis = 2))
                        filter_funcs.append(List)
                elif filt == 'Izz':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 0),Smooth,axis = 1)\
                                            ,Smooth,axis = 2))
                        filter_funcs.append(List)
                elif filt == 'Iyz':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 0),h,axis = 2)\
                                            ,Smooth,axis = 0))
                        filter_funcs.append(List)
                elif filt == 'Iyy':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 2),Smooth,axis = 1)\
                                            ,Smooth,axis = 0))
                        filter_funcs.append(List)
                elif filt == 'Ixz':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 1),Smooth,axis = 2)\
                                            ,h,axis = 0))
                        filter_funcs.append(List)
                elif filt == 'Ixy':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 1),h,axis = 2)\
                                            ,Smooth,axis = 0))
                        filter_funcs.append(List)
                elif filt == 'Ixx':
                    List = []
                    for j in range(data.shape[3]):
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,k=j,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(convolve1d(I[:,:,:,k],h,axis = 1),Smooth,axis = 2)\
                                            ,Smooth,axis = 0))
                        filter_funcs.append(List)
                elif filt == 'M':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_masks[ind_mask]: 
                                convolve(I[:,:,:,k],h) )
                    ind_mask += 1
                elif filt == '|Ix|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial(p_filter,1,0,0): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Iy|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,0,0,1): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Iz|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,0,1,0): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|gradI|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I, k=j, hx=filter_binomial_3D(p_filter,1,0,0), 
                            hy=filter_binomial_3D(p_filter,0,0,1),hz = filter_binomial_3D(p_filter,0,1,0):
                                np.sqrt(convolve(I[:,:,:,k],hx)**2 
                                        + convolve(I[:,:,:,k],hy)**2+convolve(I[:,:,:,k],hz)**2) )
                elif filt == '|Ixx|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,2,0,0): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Ixy|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial(p_filter,1,0,1): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Iyy|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial(p_filter,0,0,2): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Ixz|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,1,1,0): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Iyz|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,0,1,1): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|Izz|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_binomial_3D(p_filter,0,2,0): 
                                np.abs(convolve(I[:,:,:,k],h)) )
                elif filt == '|M|':
                    for j in range(data.shape[3]):
                        filter_funcs.append(
                            lambda I,k=j,h=filter_masks[ind_mask]: 
                                np.abs(convolve(I[:,:,:,k],h)) )
                    ind_mask += 1
                else:
                    print('Unknown filter!')
                # apply filters to image
            f_vec = np.empty((data.shape[0],data.shape[1],data.shape[2],len(self.filter_list)), 
                                dtype=data.dtype)
            for i in range(0,f_vec.shape[3]):
                Response = []
                for filter_scale in filter_funcs[i]:
                    Response.append(filter_scale(data))
                f_vec[:,:,:,i] = np.array(Response).max(0)
        else:
            if channel_num == 1:
                filter_funcs = []
                ind_mask = 0
                for filt in self.filter_list:
                    if filt == 'x':
                        filter_funcs.append([
                            lambda I,res=ind[1][::-1]: ind[1][::-1]/I.shape[1]])
                    elif filt == 'z':
                        filter_funcs.append(
                            [lambda I,res=ind[0][::-1]: ind[0][::-1]/I.shape[0]])
                    elif filt == 'I0':
                        filter_funcs.append([lambda I: I[:,:] ])
                    elif filt == 'I':
                        filter_funcs.append([
                            lambda I,h=filter_binomial(p_filter,0,0): 
                                    convolve(I[:,:],h)] )
                    elif filt == 'Ix':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(I[:,:],h,axis = 1),Smooth,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'Iz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(I[:,:],h,axis = 0),Smooth,axis = 1))
                        filter_funcs.append(List)
                    elif filt == 'Izz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(I[:,:],h,axis = 0),Smooth,axis = 1))

                        filter_funcs.append(List)
                    elif filt == 'Ixz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(I[:,:],h,axis = 1),h,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'Ixx':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(I[:,:],h,axis = 1),h,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'M':
                        filter_funcs.append(
                            lambda I,h=filter_masks[ind_mask]: 
                                convolve(I[:,:],h) )
                        ind_mask += 1
                    elif filt == '|Ix|':
                        filter_funcs.append(lambda I,h=filter_binomial(p_filter,1,0,0): 
                                np.abs(convolve(I[:,:],h)) )
                    elif filt == '|Iz|':
                        filter_funcs.append(
                            lambda I,h=filter_binomial_3D(p_filter,0,1,0): 
                                np.abs(convolve(I[:,:],h)) )
                    elif filt == '|gradI|':
                        filter_funcs.append(
                            lambda I, hx=filter_binomial_3D(p_filter,1,0,0),hz = filter_binomial_3D(p_filter,0,1,0):
                                np.sqrt(convolve(I[:,:],hx)**2+convolve(I[:,:],hz)**2) )
                    elif filt == '|Ixx|':
                        filter_funcs.append(
                            lambda I,h=filter_binomial(p_filter,2,0,0): 
                                np.abs(convolve(I[:,:],h)) )
                    elif filt == '|Ixz|':
                        filter_funcs.append(
                            lambda I,h=filter_binomial(p_filter,1,1,0): 
                                np.abs(convolve(I[:,:],h)) )
                    elif filt == '|Izz|':
                        filter_funcs.append(
                            lambda I,h=filter_binomial(p_filter,0,2,0): 
                                np.abs(convolve(I[:,:],h)) )
                    elif filt == '|M|':
                        filter_funcs.append(
                            lambda I,h=filter_masks[ind_mask]: 
                                np.abs(convolve(I[:,:],h)) )
                        ind_mask += 1
                    else:
                        print('Unknown filter!')
                # apply filters to image
                f_vec = np.empty((data.shape[0],data.shape[1],len(self.filter_list)), 
                                    dtype=data.dtype)
            else:
                filter_funcs = []
                ind_mask = 0
                for filt in self.filter_list:
                    if filt == 'x':
                        filter_funcs.append([lambda I,res=ind[1][::-1]: ind[1][::-1]/I.shape[1]])
                    elif filt == 'z':
                        filter_funcs.append([lambda I,res=ind[0][::-1]: ind[0][::-1]/I.shape[0]])
                    elif filt == 'H':
                        filter_funcs.append([lambda I: rgb2hsv(I)[:,:,0] ])
                    elif filt == 'S':
                        filter_funcs.append([lambda I: rgb2hsv(I)[:,:,1] ])
                    elif filt == 'V':
                        filter_funcs.append([lambda I: rgb2hsv(I)[:,:,2] ])
                    elif filt == 'I':
                        filter_funcs.append([
                            lambda I,h=filter_binomial(p_filter,0,0): 
                                    convolve(np.einsum('ijk,k->ij',I[:,:,:],mask),h)] )
                    elif filt == 'Ix':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),Smooth,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'Iz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 0),Smooth,axis = 1))
                        filter_funcs.append(List)
                    elif filt == 'Izz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 0),Smooth,axis = 1))

                        filter_funcs.append(List)
                    elif filt == 'Ixz':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),h,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'Ixx':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),Smooth,axis = 0))
                        filter_funcs.append(List)
                    elif filt == 'M':
                        filter_funcs.append(
                            lambda I,h=filter_masks[ind_mask]: 
                                convolve(I[:,:],h) )
                        ind_mask += 1
                    elif filt == '|Ix|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    np.abs(convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),Smooth,axis = 0)))
                        filter_funcs.append(List)
                    elif filt == '|Iz|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    np.abs(convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 0),Smooth,axis = 1)))
                        filter_funcs.append(List)
                    elif filt == '|gradI|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    np.sqrt( convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 0)**2+convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1)**2)) 
                        filter_funcs.append(List)
                    elif filt == '|Ixx|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    np.abs(convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),Smooth,axis = 0)))
                        filter_funcs.append(List)
                    elif filt == '|Ixz|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,1): 
                                    np.abs(convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 1),h,axis = 0)))
                        filter_funcs.append(List)
                    elif filt == '|Izz|':
                        List = []
                        Smooth = filter_binomial1d(3,0)
                        for size in Scale_List:
                            List.append(
                                lambda I,h=filter_binomial1d(size,2): 
                                    np.abs(convolve1d(convolve1d(np.einsum('ijk,k->ij',I[:,:,:],mask),h,axis = 0),Smooth,axis = 1)))
                        filter_funcs.append(List)
                    elif filt == '|M|':
                        filter_funcs.append(
                            lambda I,h=filter_masks[ind_mask]: 
                                np.abs(convolve(np.einsum('ijk,k->ij',I[:,:,:],mask),h)) )
                        ind_mask += 1
                    elif filt == 'E':
                        filter_funcs.append([lambda I: self.Entropy_Filter(np.einsum('ijk,k->ij',I[:,:,:],mask))])
                    elif filt == 'W':
                        pass
                    else:
                        print('Unknown filter!')
                # apply filters to image
                if 'W' in self.filter_list:
                    f_vec = np.empty((data.shape[0],data.shape[1],len(self.filter_list)+3), 
                                        dtype=data.dtype)
                    Curvature_List = self.Weingarten_Map(np.einsum('ijk,k->ij',data[:,:,:],mask))
                    for i in range(0,f_vec.shape[2]-4):
                        Response = []
                        for filter_scale in filter_funcs[i]:
                            Response.append(filter_scale(data[:,:,:])/(filter_scale(data[:,:,:]).max()))
                        f_vec[:,:,i] = np.squeeze(np.array(Response).max(0))
                    f_vec[:,:,i+1] = Curvature_List[0]
                    f_vec[:,:,i+2] = Curvature_List[1]
                    f_vec[:,:,i+3] = Curvature_List[2]
                    f_vec[:,:,i+4] = Curvature_List[3]
                else:            
                    f_vec = np.empty((data.shape[0],data.shape[1],len(self.filter_list)), 
                                        dtype=data.dtype)
                    for i in range(0,f_vec.shape[2]):
                        Response = []
                        for filter_scale in filter_funcs[i]:
                            Response.append(filter_scale(data[:,:,:])/(filter_scale(data[:,:,:]).max()))
                        f_vec[:,:,i] = np.squeeze(np.array(Response).max(0))
                    
        return f_vec
    
    def F_vec_to_Cov(self,p_cov = 5):
        # compute covariance matrices of feature vectors and add 
        # eps * identity matrix as regularization:
        if self.HDim == True:
            cov = self.covariance_from_fvec(weights = filter_binomial_3D(p_cov,0,0,0))
        else:
            cov = self.covariance_from_fvec(weights = filter_binomial(p_cov,0,0,))
        ## Postconvolution: PD matrices
        nfeatures = self.f_vec.shape[-1]
        for i in range(nfeatures):
            for j in range(nfeatures):
                gaussian(cov[:,:,:, i, j], sigma=1, order=0,
                            output=cov[:, :,:,  i, j], mode='reflect')
        
        cov = cov/(np.max(np.max(cov,axis = 3),axis = 3)[:,:,:,None,None])
        return cov
    
def filter_binomial1d(n, k):
        """
        One-dimensional Difference of binomial (DoG) filter of size n for the 
        derivative of order k.
        """
        # sanity checks:
        assert isinstance(n, int) and n%2 == 1 and n >= 3
        assert isinstance(k, int) and k >= 0 and k <= n
        return np.array(
            [sum([(-1)**j*binom(n-1-k,i-j)*binom(k,j) for j in range(k+1)])
            *0.5**(n-1-k) for i in range(n)])



def filter_binomial(n, kx, ky):
    """
    Two-dimensional binomial (DoG) filter of size (n,n) for the derivative 
    of order (kx,ky).
    """
    return np.outer(filter_binomial1d(n,ky), filter_binomial1d(n,kx))

def filter_binomial_3D(n, kx, ky,kz):
    """
    Three-dimensional binomial (DoG) filter of size (n,n,n) for the derivative 
    of order (kx,ky,kz).
    """
    return np.einsum('x,y,z-> yxz',filter_binomial1d(n,kx),filter_binomial1d(n,ky),filter_binomial1d(n,kz))
