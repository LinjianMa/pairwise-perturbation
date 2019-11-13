#ifndef __ALS_CP3_H__
#define __ALS_CP3_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

bool alscp_dt3(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, double tol,
               double timelimit, int maxiter, double lambda,
               ofstream &Plot_File, int resprint, bool bench, World &dw);

#endif
