all:
	cd Cython; \
	python setup.py build_ext --inplace
	rm -f -r Cython/build
	rm -f Cython/Geometric_*.c
	rm -f Cython/cython_*.c
	rm -f Cython/Graph*.c
clean:
	rm -f Cython/Geometric*.so
	rm -f Cython/cython_*.so
	rm -f Cython/Graph*.so
