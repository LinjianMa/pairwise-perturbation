#ifndef __ALS_CP3_H__
#define __ALS_CP3_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

bool alscp_dt3(Tensor<> &V, Matrix<> *W, int maxiter, double lambda,
               ofstream &Plot_File, int resprint, int partition, World &dw);

bool alscp_pp3(Tensor<> &V, Matrix<> *W, int maxiter, double pp_res_tol,
               double lambda, ofstream &Plot_File, int resprint, int partition, World &dw);

void alscp_dt3_sub(Tensor<> &V, Matrix<> *W, Matrix<> *dW, double tol_init,
                   int maxiter, double &st_time, double lambda,
                   ofstream &Plot_File, int &iter, int resprint, World &dw);

void alscp_pp3_sub(Tensor<> &V, Matrix<> *W, Matrix<> *dW, double tol_init,
                   int maxiter, double &st_time, double lambda,
                   ofstream &Plot_File, int &iter, int resprint, int partition, World &dw);

#endif
