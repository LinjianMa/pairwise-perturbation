#main compiler to use
CXX=mpicxx
#compiler flags (-g -O0 for debug, -O3 for optimization), generally need -fopenmp and -std=c++0x
CXXFLAGS= -O0 -std=c++0x -DOMP_OFF -Wall -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -D_DARWIN_C_SOURCE -DFTN_UNDERSCORE=1 -DUSE_LAPACK -DUSE_SCALAPACK 
#path to CTF include directory prefixed with -I
INCLUDE_PATH=-I/Users/linjian/Documents/ctf/include
#path to MPI/CTF/scalapack/lapack/blas library directories prefixed with -L
LIB_PATH= -L/Users/linjian/Documents/ctf/scalapack/build/lib  -L/Users/linjian/Documents/ctf/lib -L/Users/linjian/Documents/ctf/lib_shared
#libraries to link (MPI/CTF/scalapack/lapack/blas) to prefixed with -l
LIBS=-lctf -lscalapack -llapack -lblas
