#ifndef __CP_MSDT_LR_OPTIMIZER_H__
#define __CP_MSDT_LR_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>
class CPMSDTLROptimizer : public CPOptimizer<dtype> {

	public:
		CPMSDTLROptimizer(int order, int r, World & dw);

		~CPMSDTLROptimizer();

		void step();

		void update_indexes();

		void Construct_Dimension_Tree();

		void Construct_Subtree(vector<int> top_node);

		void Right_Subtree(vector<int> top_node);

		void mttkrp_map_init(int left_index);

		void mttkrp_map_DT(string index);

		void vec2str(vector<int> vec, string & seq);

		void update_cached_tensor(int index);

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

		// for low rank update
		// either specify the rank or tolerance
		int rank;
		bool low_rank_decomp;
		bool* is_cached;
		Tensor<>* cached_tensors;
		Matrix<> U;
		Vector<> s;
		Matrix<> VT;
};


#include "cp_msdt_lr_optimizer.cxx"

#endif
