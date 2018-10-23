#main compiler to use
CXX=mpicxx
#compiler flags (-g -O0 for debug, -O3 for optimization), generally need -fopenmp and -std=c++0x
CXXFLAGS=-std=c++0x -g -O0 -fopenmp
#path to CTF include directory prefixed with -I
INCLUDE_PATH=
#path to MPI/CTF/scalapack/lapack/blas library directories prefixed with -L
LIB_PATH=
#libraries to link (MPI/CTF/scalapack/lapack/blas) to prefixed with -l
LIBS=-lctf
