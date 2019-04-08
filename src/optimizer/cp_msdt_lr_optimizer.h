#ifndef __CP_MSDT_LR_OPTIMIZER_H__
#define __CP_MSDT_LR_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>
class CPMSDTLROptimizer : public CPMSDTOptimizer<dtype> {

    public:
        CPMSDTLROptimizer(int order, int r, int update_rank, int randomsvd, World & dw);

        ~CPMSDTLROptimizer();

        double step();

        void mttkrp_map_init(int left_index);

        void mttkrp_map_DT(string index);

        void update_cached_tensor(int index);

        // for low rank update
        // either specify the rank or tolerance
        int rank;
        bool low_rank_decomp;
        bool* is_cached;
        Tensor<>* cached_tensors;
        Matrix<>* old_W;
        Matrix<> U;
        Vector<> s;
        Matrix<> VT;
        bool randomsvd;
};


#include "cp_msdt_lr_optimizer.cxx"

#endif
