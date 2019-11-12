#ifndef __CP_MSDT_OPTIMIZER_H__
#define __CP_MSDT_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPMSDTOptimizer : public CPOptimizer<dtype> {

public:
  CPMSDTOptimizer(int order, int r, World &dw);

  ~CPMSDTOptimizer();

  double step();

  void update_indexes();

  void Construct_Dimension_Tree();

  void Construct_Subtree(vector<int> top_node);

  void Right_Subtree(vector<int> top_node);

  void mttkrp_map_init(int left_index);

  void mttkrp_map_DT(string index);

  char seq_V[100];
  // used for doing the first contraction
  // mttkrp_map[seq][seq_map_init] = V[seq_V] * W[i][seq2]
  char seq_map_init[100];
  // used for building the MSDT.
  // sub of seq_V
  char seq_tree_top[100];

  // maps
  map<string, Tensor<dtype>> mttkrp_map;
  map<string, string> parent;
  map<string, string> contract_index;

  // indices that update in one step
  vector<int> indexes;
  int left_index;
};

#include "cp_msdt_optimizer.cxx"

#endif
