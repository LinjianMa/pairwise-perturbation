#ifndef __CP_DT_OPTIMIZER_H__
#define __CP_DT_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPDTOptimizer : public CPOptimizer<dtype> {

public:
  CPDTOptimizer(int order, int r, World &dw);

  ~CPDTOptimizer();

  double step();

  void update_left_index();

  void update_indexes(vector<int> &indexes, int left_index);

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
  bool first_subtree;
  vector<int> indexes;
  vector<int> indexes1;
  vector<int> indexes2;
  int left_index;
  int left_index1;
  int left_index2;
  int special_index;
};

#include "cp_dt_optimizer.cxx"

#endif
