# Local_Feature_Extraction_Lib
Library for extracting local features, prototypes on Curved Manifold of Symmetric Positive Definite Matrices (SPD) for low level image partitioning and distance computation. 

## Partitioning of color images via different metrics (postively, negatily and flat) curved manifolds. 
Supports Riemannian means including 
   - Log-Euclidian Means using Euclidian Metrics proposed in paper: 
   - (Cheap Mean)
   - (Bini Mean)
   - Riamannian Mean
   Comparison of key properties satisfied by the mean with respect to implemented metrics:

   ![titleimageA](/docs/Mean_Prop.png)
   
   Left: natural image; Right: partitioning using the piecewise affine-linear Mumford-Shah model
   
   - Avoids oversegmentation of images with linear trends (e.g. the sky in a landscape image, illumination gradients)
   
   ![PottsAndPALMS](/docs/Vis_Deep.png)
   ![PottsAndPALMS](/docs/Deep_Cov_Vis.png)
   
   Left: natural image; Center: classical (piecewise constant) Potts model; Right: piecewise affine-linear Mumford-Shah model
   
   - Scale of the partitioning is controlled by a model parameter
   
   ![parameter](/docs/parameter.png)
   
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

# `f_vec` contains the feature vectors
# `cov_matrices` contains the covariance matrices for each pixel


## Installation
### Compiling
The algorithm depends on a mex script that needs to be compiled before execution. For compilation inside MATLAB, cd into the 'src/cpp' folder and run build.m

Requires the Armadillo and OpenMP library

Tested with Armadillo 8.400 https://launchpad.net/ubuntu/+source/armadillo/1:8.400.0+dfsg-2 and OpenMP 4.0.
On Linux, just use your package manager to install it:

sudo apt-get install libarmadillo-dev

sudo apt-get install libomp-dev

### Running
For a test run on the test image "redMacaw", run demo.m. 
demo.m calls the main function, affineLinearPartitioning.m
Arguments of affineLinearPartitioning.m are:
 - f: input image (double)
 - gamma: boundary penalty (larger choice -> less segments)
 - varargin: optional input parameters

## References
- L. Kiefer, M. Storath, A. Weinmann.
    "An efficient algorithm for the piecewise affine-linear Mumford-Shah model based on a Taylor jet splitting."
    IEEE Transactions on Image Processing, 2020.
- L. Kiefer, M. Storath, A. Weinmann.
    "PALMS image partitioning – a new parallel algorithm for the piecewise affine-linear Mumford-Shah model."
     Image Processing On Line, 2020.