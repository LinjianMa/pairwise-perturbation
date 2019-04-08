#ifndef __CP_H__
#define __CP_H__

#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template<typename dtype, class Optimizer>  
class CPD : public Decomposition<dtype> {

    public:

        // grad_W: gradient in each dimension
        Matrix<> * grad_W = NULL;
        double gradnorm = 0.;
        Optimizer * optimizer = NULL;

        CPD(int order, int size, int r, World & dw);

        CPD(int order, int size, int r, int update_rank, World & dw);

        CPD(int order, int size, int r, int update_rank, int randomsvd, World & dw);

        CPD(int order, int* size, int* r, World & dw);

        void Init(Tensor<dtype>* input, Matrix<dtype>* mat, double lambda=0.);

        ~CPD();

        void print_grad(int i) const;

        void update_gradnorm();

        /**
         * \brief ALS method for CP decomposition
         *  tol: tolerance for a relative stopping condition
         *  timelimit, maxiter: limit of time and iterations
         */
        bool als(double tol, double timelimit, int maxsweep, int resprint, ofstream & Plot_File, bool bench = false);

        char seq_V[100];

};

#include "CP.cxx"

#endif
