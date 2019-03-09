#ifndef __SIMPLE_OPTIMIZER_H__
#define __SIMPLE_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>  
class SimpleOptimizer : public Optimizer<dtype> {

	public:
		SimpleOptimizer(int order, int r, World & dw);

		~SimpleOptimizer();

		void step();

		char seq_V[100];

};


#include "simple_optimizer.cxx"

#endif

