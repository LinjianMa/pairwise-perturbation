#ifndef __DECOMPOSITION_H__
#define __DECOMPOSITION_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class Decomposition {

public:
  Decomposition(int order, int size, int r, World &dw);

  Decomposition(int order, int *size, int *r, World &dw);

  // Decomposition(const Decomposition & other);

  ~Decomposition();

  // Decomposition<dtype> & operator=(Decomposition<dtype> const & other);

  void Init(Tensor<dtype> *input, Matrix<dtype> *mat);

  void print_V() const;

  void print_W(int i) const;

  // V: input tensor
  Tensor<dtype> *V = NULL;
  int order;
  int *size;
  int *rank;
  // W: output solutions
  Matrix<dtype> *W = NULL;
  // DimensionTree<dtype>* dtree_;
  World *world;
};

#include "decomposition.cxx"

#endif
