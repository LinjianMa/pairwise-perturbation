include config.mk

FCXX=$(CXX) $(CXXFLAGS)

all: pp_bench

pp_bench: pp_bench.cxx als_CP.o als_Tucker.o common.o Makefile config.mk
	$(FCXX) $< als_CP.o als_Tucker.o common.o  -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

als_CP.o: als_CP.cxx als_CP.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

als_Tucker.o: als_Tucker.cxx als_Tucker.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

common.o: common.cxx common.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

clean:
	rm -f common.o als_CP.o als_Tucker.o pp_bench
