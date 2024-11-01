import numpy as np
from scipy.ndimage.filters import convolve, convolve1d
from scipy.special import binom
from scipy.ndimage.filters import gaussian_filter as gaussian

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

All filters, except for 'x' and 'y', are applied channel-wise. For example, for an RGB image, applying the filters 'x', 'y', 'I', 'Ix', 'Iy' results in feature vectors of the form: `x, y, R, G, B, Rx, Gx, Bx, Ry, Gy, By`. The derivatives are computed using a binomial filter with size `(p_filter, p_filter)`.

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
    def __init__(self,data,chanels_num ,Scale_List = [3,5,7,11,15],subtract_mean = True,Hdim = False,filter_list = 'x,z,I0,I,Ix,Iz,Ixx,Ixz,Izz'):
        self.data = data
        self.subtract_mean = subtract_mean
        self.chanels_num = chanels_num 
        self.filter_list = filter_list
        self.HDim = Hdim
        self.Scale_List = Scale_List # Max Response of filter outputs over scales
        self.f_vec = self.covariance_descriptor_3D(data,Scale_List,subtract_mean)

    def covariance_from_fvec(self,weights):
        """
        Subroutine for computing covariance matrices for given feature vectors.
        """
        if self.chanels_num == 1:
            if self.HDim == True:
                z,x,y,f = self.f_vec.shape
                res = np.empty((z,x,y,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    pz,px,py = weights.shape
                    weights_uniform = np.ones((pz,px,py)) / (px*py*pz)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for i in range(f):
                        fvec_mean[:,:,:,i] = convolve(self.f_vec[:,:,:,i], weights_uniform)
                        fvec_mean_w[:,:,:,i] = convolve(self.f_vec[:,:,:,i], weights) 
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,i,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], weights) \
                                - fvec_mean[:,:,:,i] * fvec_mean_w[:,:,:,j] \
                                - fvec_mean_w[:,:,:,i] * fvec_mean[:,:,:,j] \
                                + fvec_mean[:,:,:,i] * fvec_mean[:,:,:,j] * np.sum(weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,i,j] = convolve(self.f_vec[:,:,:,i] * self.f_vec[:,:,:,j], weights)
                            res[:,:,:,j,i] = res[:,:,:,i,j]
            else:
                m,n,f = self.f_vec.shape
                res = np.empty((m,n,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    px,py = weights.shape
                    weights_uniform = np.ones((px,py)) / (px*py)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for i in range(f):
                        fvec_mean[:,:,i] = convolve(self.f_vec[:,:,i], weights_uniform)
                        fvec_mean_w[:,:,i] = convolve(self.f_vec[:,:,i], weights) 
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], weights) \
                                - fvec_mean[:,:,i] * fvec_mean_w[:,:,j] \
                                - fvec_mean_w[:,:,i] * fvec_mean[:,:,j] \
                                + fvec_mean[:,:,i] * fvec_mean[:,:,j] * np.sum(weights)
                            res[:,:,j,i] = res[:,:,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,i,j] = convolve(self.f_vec[:,:,i] * self.f_vec[:,:,j], weights)
                            res[:,:,j,i] = res[:,:,i,j]
        else:
            if self.HDim == True:
                z,x,y,channels,f = self.f_vec.shape
                res = np.empty((z,x,y,channals,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    pz,px,py = weights.shape
                    weights_uniform = np.ones((pz,px,py)) / (px*py*pz)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for chan in range(channels):
                        for i in range(f):
                            fvec_mean[:,:,:,chan,i] = convolve(self.f_vec[:,:,:,chan,i], weights_uniform)
                            fvec_mean_w[:,:,:,chan,i] = convolve(self.f_vec[:,:,:,chan,i], weights) 
                    for chan in range(channels):
                        for i in range(f):
                            for j in range(i+1):
                                res[:,:,:,chan,i,j] = convolve(self.f_vec[:,:,:,chan,i] * self.f_vec[:,:,:,chan,j], weights) \
                                    - fvec_mean[:,:,:,chan,i] * fvec_mean_w[:,:,:,chan,j] \
                                    - fvec_mean_w[:,:,:,chan,i] * fvec_mean[:,:,:,chan,j] \
                                    + fvec_mean[:,:,:,chan,i] * fvec_mean[:,:,:,chan,j] * np.sum(weights)
                                res[:,:,:,chan,j,i] = res[:,:,:,chan,i,j]
                else:
                    for i in range(f):
                        for j in range(i+1):
                            res[:,:,:,chani,j] = convolve(self.f_vec[:,:,:,chan,i] * self.f_vec[:,:,:,chan,j], weights)
                            res[:,:,:,chan,j,i] = res[:,:,:,chan,i,j]
            else:
                m,n,channels,f = self.f_vec.shape
                res = np.empty((m,n,channels,f,f), dtype=self.f_vec.dtype)  # variable for the result
                if self.subtract_mean:
                    px,py = weights.shape
                    weights_uniform = np.ones((px,py)) / (px*py)
                    fvec_mean = np.empty_like(self.f_vec) # (unweighted) mean of self.f_vec
                    fvec_mean_w = np.empty_like(self.f_vec) # weighted mean of self.f_vec
                    for chan in range(channels):
                        for i in range(f):
                            fvec_mean[:,:,chan,i] = convolve(self.f_vec[:,:,chan,i], weights_uniform)
                            fvec_mean_w[:,:,chan,i] = convolve(self.f_vec[:,:,chan,i], weights) 
                    for chan in range(channels):
                        for i in range(f):
                            for j in range(i+1):
                                res[:,:,chan,i,j] = convolve(self.f_vec[:,:,chan,i] * self.f_vec[:,:,chan,j], weights) \
                                    - fvec_mean[:,:,chan,i] * fvec_mean_w[:,:,chan,j] \
                                    - fvec_mean_w[:,:,chan,i] * fvec_mean[:,:,chan,j] \
                                    + fvec_mean[:,:,chan,i] * fvec_mean[:,:,chan,j] * np.sum(weights)
                                res[:,:,chan,j,i] = res[:,:,chan,i,j]
                else:
                    for chan in range(channels):
                        for i in range(f):
                            for j in range(i+1):
                                res[:,:,chan,i,j] = convolve(self.f_vec[:,:,chan,i] * self.f_vec[:,:,chan,j], weights)
                                res[:,:,chan,j,i] = res[:,:,chan,i,j]
        return res

    def covariance_descriptor_3D(self,data,Scale_List,subtract_mean,p_filter = 5, p_cov = 5,eps_pd = 0.0, filter_masks = [],HDim = False):
        
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
            print("HDIM")
            self.filter_list = 'x,y,z,I0,I,Ix,Iy,Iz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz'
            self.filter_list = self.filter_list.split(',')
            filter_funcs = []
            ind_mask = 0
            for filt in self.filter_list:
                if filt == 'y':
                    filter_funcs.append([lambda I,res=ind[2][::-1]: ind[2]/I.shape[2]])
                elif filt == 'x':
                    filter_funcs.append([
                        lambda I,res=ind[1][::-1]: ind[1][::-1]/I.shape[1]])
                elif filt == 'z':
                    filter_funcs.append(
                        [lambda I,res=ind[0][::-1]: ind[0][::-1]/I.shape[0]])
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
                        lambda I,h=filter_binomial_3D(p_filter,2,0,0): 
                            np.abs(convolve(I[:,:],h)) )
                elif filt == '|Ixz|':
                    filter_funcs.append(
                        lambda I,h=filter_binomial_3D(p_filter,1,1,0): 
                            np.abs(convolve(I[:,:],h)) )
                elif filt == '|Izz|':
                    filter_funcs.append(
                        lambda I,h=filter_binomial_3D(p_filter,0,2,0): 
                            np.abs(convolve(I[:,:],h)) )
                elif filt == '|M|':
                    filter_funcs.append(
                        lambda I,h=filter_masks[ind_mask]: 
                            np.abs(convolve(I[:,:],h)) )
                    ind_mask += 1
                else:
                    print('Unknown filter!')
                # apply filters to image
            f_vec = np.empty((data.shape[0],data.shape[1],data.shape[2],len(self.filter_list)), 
                                dtype=data.dtype)
            for j in range(data.shape[2]):
                for i in range(0,f_vec.shape[3]):
                    Response = []
                    for filter_scale in filter_funcs[i]:
                        Response.append(filter_scale(data[:,:,j]))
                    f_vec[:,:,j,i] = np.squeeze(np.array(Response).max(0))

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
