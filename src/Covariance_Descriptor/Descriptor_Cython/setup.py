from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name='Geometric_Mean_Utils',
        sources=['Geometric_Mean_Utils.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='cython_tools',
        sources=['cython_tools.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Graph_Tools',
        sources=['Graph_Tools.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='cython_distance',
        sources=['cython_distance.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
]

setup(
      name = 'Optimized methods',
      ext_modules = cythonize(extensions,
    compiler_directives={'language_level' : "3"})
)
