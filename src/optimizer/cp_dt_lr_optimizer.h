#ifndef __CP_DT_LR_OPTIMIZER_H__
#define __CP_DT_LR_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>
class CPDTLROptimizer : public CPDTOptimizer<dtype> {

    public:
        CPDTLROptimizer(int order, int r, int update_rank, World & dw);

        ~CPDTLROptimizer();

        double step();

        void initialize_low_rank_param();

        void update_left_index();

        void mttkrp_map_init(int left_index);

        void mttkrp_map_DT(string index);

        void update_cached_tensor(int index);

        int num_subiteration;
        int count_subiteration;
        // for low rank update
        // either specify the rank or tolerance
        int rank;
        bool low_rank_decomp;
        Tensor<>* cached_tensor1;
        Tensor<>* cached_tensor2;
        Matrix<> U;
        Vector<> s;
        Matrix<> VT;
};


#include "cp_dt_lr_optimizer.cxx"

#endif
