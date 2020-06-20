#main compiler to use
CXX=mpicxx
#compiler flags (-g -O0 for debug, -O3 for optimization), generally need -fopenmp and -std=c++0x
CXXFLAGS=-std=c++0x -O0  -fopenmp -no-ipo
#path to CTF include directory prefixed with -I
INCLUDE_PATH=-I/work/03940/lma16/stampede2/ctf-conda/include
#path to MPI/CTF/scalapack/lapack/blas library directories prefixed with -L
#LIB_PATH=-L/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64 -L/work/03940/lma16/stampede2/ctf/lib -L/work/03940/lma16/stampede2/ctf/hptt/lib
LIB_PATH=-L/home1/03940/lma16/anaconda3/lib  -L/work/03940/lma16/stampede2/ctf-conda/lib -L/work/03940/lma16/stampede2/ctf-conda/hptt/lib
#
#libraries to link (MPI/CTF/scalapack/lapack/blas) to prefixed with -l
LIBS=-lctf -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -Wl,--end-group -lpthread -lm -Wl,-Bstatic -lhptt -Wl,-Bdynamic
