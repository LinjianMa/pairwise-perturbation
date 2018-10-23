include config.mk

FCXX=$(CXX) $(CXXFLAGS)

all: test_ALS

test_ALS: test_ALS.cxx als_CP.o als_Tucker.o common.o Makefile config.mk
	$(FCXX) $< als_CP.o als_Tucker.o common.o  -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

als_CP.o: als_CP.cxx als_CP.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

als_Tucker.o: als_Tucker.cxx als_Tucker.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

common.o: common.cxx common.h 
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

clean:
	rm common.o als_CP.o als_Tucker.o test_ALS
