include config.mk

BDIR=$(shell pwd)
ODIR=$(BDIR)/obj
SDIR=$(BDIR)/src
TDIR=$(BDIR)/tests

FCXX=$(CXX) $(CXXFLAGS)

all: test_ALS pp_bench test #tool

# tool: $(ODIR)/decomposition.o #$(ODIR)/CP.o $(ODIR)/Tucker.o

# test: $(TDIR)/test_decomposition.cxx $(ODIR)/decomposition.o Makefile config.mk
# 	$(FCXX) $< $(ODIR)/decomposition.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

test: $(TDIR)/test_decomposition.cxx $(ODIR)/common.o Makefile config.mk
	$(FCXX) $< $(ODIR)/common.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

# $(ODIR)/decomposition.o: $(SDIR)/decomposition.h  config.mk
# 	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

# $(ODIR)/CP.o: $(SDIR)/CP.h $(SDIR)/CP.cxx config.mk
# 	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

# $(ODIR)/Tucker.o: $(SDIR)/Tucker.h $(SDIR)/Tucker.cxx config.mk
# 	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

pp_bench: pp_bench.cxx $(ODIR)/als_CP.o $(ODIR)/als_Tucker.o $(ODIR)/common.o Makefile config.mk
	$(FCXX) $< $(ODIR)/als_CP.o $(ODIR)/als_Tucker.o $(ODIR)/common.o  -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

test_ALS: test_ALS.cxx $(ODIR)/als_CP.o $(ODIR)/als_Tucker.o $(ODIR)/common.o Makefile config.mk
	$(FCXX) $< $(ODIR)/als_CP.o $(ODIR)/als_Tucker.o $(ODIR)/common.o  -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

$(ODIR)/als_CP.o: als_CP.cxx als_CP.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

$(ODIR)/als_Tucker.o: als_Tucker.cxx als_Tucker.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

$(ODIR)/common.o: common.cxx common.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

clean:
	rm -f $(ODIR)/*.o  test_ALS pp_bench test
