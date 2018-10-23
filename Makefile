CXX=mpicxx
CXXFLAGS=-std=c++0x -g -O0
FCXX=$(CXX) $(CXXFLAGS)
INCLUDE_PATH=
LIB_PATH=
LIBS=-lctf

all: test_ALS

test_ALS: test_ALS.cxx als_CP.o als_Tucker.o common.o Makefile 
	$(FCXX) $< als_CP.o als_Tucker.o common.o  -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

als_CP.o: als_CP.cxx als_CP.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

als_Tucker.o: als_Tucker.cxx als_Tucker.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

common.o: common.cxx common.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

