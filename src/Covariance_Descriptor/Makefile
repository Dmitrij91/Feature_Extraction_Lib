all:
	cd Descriptor_Cython; \
	python setup.py build_ext --inplace
	rm -f -r Descriptor_Cython/build
	rm -f Descriptor_Cython/Geometric_*.c
	rm -f Descriptor_Cython/cython_*.c
	rm -f Descriptor_Cython/Graph*.c
clean:
	rm -f Descriptor_Cython/Geometric*.so
	rm -f Descriptor_Cython/cython_*.so
	rm -f Descriptor_Cython/Graph*.so
