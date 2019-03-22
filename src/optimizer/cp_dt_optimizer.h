#ifndef __CP_DT_OPTIMIZER_H__
#define __CP_DT_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>  
class CPDTOptimizer : public CPOptimizer<dtype> {

    public:
        CPDTOptimizer(int order, int r, World & dw);

        ~CPDTOptimizer();

        void step();

        char seq_V[100];
        char seq[100];

        // maps 
        map<string, Tensor<dtype>> mttkrp_map;
        map<string, string> parent;
        map<string, string> sibling;

};


#include "cp_dt_optimizer.cxx"

#endif

