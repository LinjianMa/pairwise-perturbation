#ifndef __CP_SIMPLE_OPTIMIZER_H__
#define __CP_SIMPLE_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPSimpleOptimizer : public CPOptimizer<dtype> {

public:
  CPSimpleOptimizer(int order, int r, World &dw);

  ~CPSimpleOptimizer();

  double step();

  char seq_V[100];
};

#include "cp_simple_optimizer.cxx"

#endif
