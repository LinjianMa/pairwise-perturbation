
// #include "decomposition.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
Decomposition<dtype>::Decomposition(int order, int size_, int r, World &dw) {

  world = &dw;
  this->order = order;
  this->size = new int[order];
  this->rank = new int[order];
  for (int i = 0; i < order; i++) {
    this->size[i] = size_;
    this->rank[i] = r;
  }
}

template <typename dtype>
Decomposition<dtype>::Decomposition(int order, int *size_, int *r, World &dw) {

  *world = dw;
  this->order = order;
  this->size = new int[order];
  this->rank = new int[order];
  for (int i = 0; i < order; i++) {
    size[i] = size_[i];
    rank[i] = r[i];
  }
}

// template<typename dtype>
// Decomposition<dtype>::Decomposition(const Decomposition & other){
// }

template <typename dtype> Decomposition<dtype>::~Decomposition() {
  if (V != NULL) {
    // FIXME: this makes low rank MSDT crash
    // delete V;
  }
  if (W != NULL) {
    delete[] W;
  }
  delete[] size;
  delete[] rank;
}

// template<typename dtype>
// Decomposition<dtype> & Decomposition<dtype>::operator=(Decomposition<dtype>
// const & other){
// }

template <typename dtype>
void Decomposition<dtype>::Init(Tensor<dtype> *input, Matrix<dtype> *W) {
  assert(input->order == order);
  for (int i = 0; i < order; i++) {
    assert(input->lens[i] == size[i]);
    assert(W[i].ncol == rank[i]);
  }
  if (V != NULL) {
    delete V;
  }
  V = input;
  if (W != NULL) {
    delete[] this->W;
  }
  this->W = W;
}

template <typename dtype> void Decomposition<dtype>::print_V() const {
  assert(V != NULL);
  V->print();
}

template <typename dtype> void Decomposition<dtype>::print_W(int i) const {
  assert(W != NULL);
  W[i].print();
}
