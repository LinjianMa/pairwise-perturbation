
.PHONY:
$(ALS): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx  $(BDIR)/lib/libctf.a Makefile ../Makefile ../src/interface
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)


# $(BDIR)/bin/test_ALS: test_ALS.cxx als_CP.cxx als_Tucker.cxx common.cxx $(ODIR)/als_CP.o $(ODIR)/als_Tucker.o $(ODIR)/common.o $(BDIR)/lib/libctf.a Makefile ../Makefile 
# 	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)


# $(ODIR)/als_CP.o: als_CP.cxx als_CP.h ../src/interface
# 	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 

# $(ODIR)/als_Tucker.o: als_Tucker.cxx als_Tucker.h ../src/interface
# 	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 

# $(ODIR)/common.o: common.cxx common.h ../src/interface
# 	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 

