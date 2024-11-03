from numpy import allclose, asarray, empty_like
from numpy import all as np_all
from scipy.linalg import eigvalsh
import Descriptor_Cython.Geometric_Mean_Utils



def is_spd(A):
    """
    Returns True, if A is a symmetric positive definite (SPD) matrix or 
    a sequence of symmetric positive matrices, and False otherwise.
    """
    A = asarray(A)
    if A.ndim == 2:
        if A.shape[0] != A.shape[1]:
            return False
        elif not allclose(A, A.T):
            return False
        elif not np_all(eigvalsh(A) > 0.0):
            return False
        else:
            return True
    elif A.ndim == 3:
        if A.shape[-2] != A.shape[-1]:
            return False
        for a in A:
            if not allclose(a, a.T):
                return False
            elif not np_all(eigvalsh(a) > 0.0):
                return False
        return True
    else:
        return False



def sqrtmh(A, out=None):
    """
    Computes the matrix square root of a SPD matrix. If A is a sequence of
    SPD matrices, then the square root is computed for each matrix.
    """
    A = asarray(A)
    assert A.ndim == 2 or A.ndim == 3
    assert A.shape[-2] == A.shape[-1]
    assert out is None or out.shape == A.shape
    
    if out is None:
        out = empty_like(A)
    if A.ndim == 2:
        return Descriptor_Cython.Geometric_Mean_Utils.sqrtmh(A)
    elif A.ndim == 3:
        return Descriptor_Cython.Geometric_Mean_Utils.sqrtmh_seq(A, out)



def expmh(A, out=None):
    """
    Computes the matrix exponential of a symmetric matrix. If A is a 
    sequence of symmetric matrices, then the exponential is computed for 
    each matrix.
    """
    A = asarray(A)
    assert A.ndim == 2 or A.ndim == 3
    assert A.shape[-2] == A.shape[-1]
    assert out is None or out.shape == A.shape
    
    if out is None:
        out = empty_like(A)
    if A.ndim == 2:
        return Descriptor_Cython.Geometric_Mean_Utils.expmh(A, out)
    elif A.ndim == 3:
        return Descriptor_Cython.Geometric_Mean_Utils.expmh_seq(A, out)


def logmh(A, out=None):
    """
    Computes the matrix logarithm of a SPD matrix. If A is a sequence of
    SPD matrices, then the logarithm is computed for each matrix.
    """
    A = asarray(A)
    assert A.ndim == 2 or A.ndim == 3
    assert A.shape[-2] == A.shape[-1]
    assert out is None or out.shape == A.shape
    
    if out is None:
        out = empty_like(A)
    if A.ndim == 2:
        return Descriptor_Cython.Geometric_Mean_Utils.logmh(A)
    elif A.ndim == 3:
        return Descriptor_Cython.Geometric_Mean_Utils.logmh_seq(A, out)



def invh(A, out=None):
    """
    Computes the matrix inveres of a SPD matrix. If A is a sequence of
    SPD matrices, then the inverse is computed for each matrix.
    """
    A = asarray(A)
    assert A.ndim == 2 or A.ndim == 3
    assert A.shape[-2] == A.shape[-1]
    assert out is None or out.shape == A.shape
    
    if out is None:
        out = empty_like(A)
    if A.ndim == 2:
        return Descriptor_Cython.Geometric_Mean_Utils.invh(A)
    elif A.ndim == 3:
        return Descriptor_Cython.Geometric_Mean_Utils.invh_seq(A, out)









