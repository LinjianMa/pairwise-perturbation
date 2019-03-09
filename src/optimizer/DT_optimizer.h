#ifndef __DT_OPTIMIZER_H__
#define __DT_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>  
class DTOptimizer : public Optimizer<dtype> {

	public:
		DTOptimizer(int order, int r, World & dw);

		~DTOptimizer();

		void step();

		char seq_V[100];
		char seq[100];

		// maps 
		map<string, Tensor<dtype>> mttkrp_map;
		map<string, string> parent;
		map<string, string> sibling;

};


#include "DT_optimizer.cxx"

#endif

