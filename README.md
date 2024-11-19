# Local Features Extraction Lib
Library for extracting local features, prototypes on Curved Manifold of Symmetric Positive Definite Matrices (SPD) for low level image partitioning and distance computation. 

![Image of horse at different levels of pixelation and detail](/docs/Horse.png)

## Partitioning of color images via different metrics (postively, negatily and flat) curved manifolds. 
Supports Riemannian means including 
   - **Log-Euclidian Means**
   - **Cheap Mean**
   - **Bini Mean**
   - **Approximate Joint Diagonalization (AJD)**
   - **Riamannian Mean**
   - **Stein Divergence Mean**
   Comparison of key properties satisfied by the mean with respect to implemented metrics:

   ![titleimageA](/docs/Mean_Prop.png)
   
   Left: Illustration of different geometric mean properties on the manifold of positive semidefinite matrices implemented in this library.  

![titleimageA](/docs/Cluster_Prototypes.png)

Deviation of the geometric means computed using the Log-Euclidian metric and Stein diver-
gence, respectively, from the true Riemannian mean. Right: Runtime for geometric mean computation using
the different metrics. All evaluations were performed on a randomly chosen subset of covariance descriptors
representing the retinal nerve fibre layer in a real-world OCT scan. This highlights the advan-
tages of using Stein the divergence in terms of approximation accuracy and efficient numerical computation.


# Deep Features Extractor

![titleimageA](/docs/Deep_Features.png)

CNN architecture for feature extraction which
comprises four residually connected blocks. For segmenting OCT-real valued data, the network produces
a sequence of hidden states with channel dimen-
sions 8, 16, 32, 64. Each block contains 3D con-
volution with filter size 3 × 3 × 3 and rectified
linear unit (ReLU) is used as activation func-
tion.

  # Detailed Breakdown of the Code for Computing Covariance Descriptors for Image Data

The code is designed to compute covariance descriptors for image data using a variety of predefined filters. Here's a detailed breakdown of key sections and functionalities of this code:

## 1. Class Initialization and Input Data

The `Features` class constructor initializes the data and sets default parameters such as:

- `self.data`: the input image data.
- `self.f_vec`: stores the feature vectors.
- `self.weights`: weights applied to compute weighted means and covariances.
- `self.Scale_List`: contains different scales to apply to the filter derivatives, controlling the scale of the filters being applied to the input data.

## 2. Covariance Matrix Computation (`covariance_from_fvec`)

This function computes covariance matrices for the feature vectors, which can be either in 2D or 3D format (controlled by `HDim`). The covariance can be weighted or unweighted:

- **Weighted**: If `subtract_mean` is `True`, it calculates both weighted and unweighted means of the feature vectors and then computes the covariance matrix by considering these means.
- **Unweighted**: If `subtract_mean` is `False`, it directly computes the covariance matrix by applying convolution filters.

The function uses the SciPy `convolve` function to apply convolutions across feature vectors to compute the covariance terms.

## 3. Filter Application (`covariance_descriptor_3D`)

The core function of the class is `covariance_descriptor_3D`, which applies various predefined filters to the input image data, then computes covariance descriptors. Key parameters include:

- `p_filter`: controls the size of the filters for derivative computations.
- `p_cov`: the size of the neighborhood used for covariance matrix computation.
- `filter_list`: a list of filters that will be applied to extract features.
- `eps_pd`: regularization parameter to ensure positive definiteness in the covariance matrices.
- `subtract_mean`: indicates whether to subtract the mean of the feature vectors before computing covariance.
- `weights`: used for applying weighted convolutions.
- `HDim`: controls whether the input is 2D or 3D.

## 4. Filters and Feature Extraction

There are several filters defined:

- **Coordinate filters (x, y, z)**: Normalize the pixel coordinates to [0, 1].
- **Intensity filters (I0, I)**: Intensity values at pixels or Gaussian-smoothed intensity values.
- **First and second derivatives (Ix, Iy, Ixx, Ixy, etc.)**: Compute first and second derivatives of intensity using binomial filters.
- **Gradient-based filters (|Ix|, |Iy|, |gradI|, etc.)**: Compute gradient magnitudes and other related features.
- **Custom filters (M)**: Users can apply custom filters via `filter_masks`.

## 5. Covariance Descriptor Output

Once the filters are applied, the covariance descriptor for each pixel is computed, regularized (to ensure positive definiteness), and smoothed using a Gaussian filter.

## 6. Helper Functions

There are several helper functions for filter creation:

- `filter_binomial1d(n, k)`: Creates a one-dimensional binomial filter for derivatives.
- `filter_binomial(n, kx, ky)`: Creates a two-dimensional binomial filter for derivatives.
- `filter_binomial_3D(n, kx, ky, kz)`: Creates a three-dimensional binomial filter.

## Example Use Case

To extract features and compute the covariance matrix for a 3D image using predefined filters:

```python
# Assuming 'image_data' is the input data, e.g., a 3D volume (m, n, p)
# Initialize the Features class with image data
features = Features(image_data)
# Apply filters and compute covariance descriptors
f_vec, cov_matrices = features.covariance_descriptor_3D(
    p_filter=5, 
    p_cov=5, 
    filter_list=['x', 'y', 'I', 'Ix', 'Iy'],  # Example filters
    eps_pd=0.01, 
    filter_masks=[],  # No custom masks provided in this case
    subtract_mean=True, 
    weights='Gaussian', 
    HDim=True  # Assuming 3D data
)
```

- `f_vec` contains the feature vectors
- `cov_matrices` contains the covariance matrices for each pixel


## Installation
### Compiling
The library depends on accelerated helper function for mean computation in cython that needs to be compiled before execution. For compilation, cd into the 'src/Covariance_Descritor/' folder and run make all

Requires the OpenMP library

On Linux, just use your package manager to install it:

sudo apt-get install libomp-dev

### Running
For a test run on the horse data set, (todo add path). 
To compuate the covariance descriptors, call the function (todo add function) with the arguments
Arguments of affineLinearPartitioning.m are:
 - f: input image (double)
 - gamma: boundary penalty (larger choice -> less segments)
 - varargin: optional input parameters

# Medical Applications Scenarios on example of volumetric Optical-Coherance-Tomography data 

![titleimageA](/docs/Segment_B_Scan.png)
Labeled retina tissues using covariance descriptors
and deep CNN features, respectively. The resulting segmentation obtained using the IOWA reference algorithm.

![titleimageA](/docs/Segment_Volume.png)

## References
- Dmitrij Sitenko, Bastian Boll, and Christoph Schnörr,
    " Assignment Flow for Order-Constrained OCT Segmentation."
    International Journal of Computer Vision 129, 11 (Nov 2021), 3088–3118.
- Dario A. Bini, Bruno Iannazzo,
    "A note on computing matrix geometric means."
    Adv Comput Math 35, 175–192 (2011). https://doi.org/10.1007/s10444-010-9165-0
- Dario A. Bini, Bruno Iannazzo,
    "Computing the Karcher mean of symmetric positive definite matrices."
    Linear Algebra and its Applications,Volume 438, Issue 4,2013