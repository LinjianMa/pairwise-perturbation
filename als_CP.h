#ifndef __ALS_CP_H__
#define __ALS_CP_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

/**
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *  F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, Matrix<> *F, double tol,
           double timelimit, int maxiter, World &dw);

/**
 * \brief ALS method for CP decomposition with decision tree
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *  F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 *  V.order should be >=4
 */
bool alsCP_DT(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, Matrix<> *F,
              double tol, double timelimit, int maxiter, double lambda,
              ofstream &Plot_File, int resprint, bool bench, World &dw);

// [cd] --> [ab*]
void stringbuilder_mttkrp(char *seq, char *seq_return, int N, World &dw);

void Build_mttkrp_map(map<string, Tensor<>> &mttkrp_map, Tensor<> &V,
                      Matrix<> *W, char *seq, World &dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *  F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
// bool alsCP_mod(Tensor<> & V,
//          Matrix<> * W,
//          Matrix<> * grad_W,
//          Matrix<> * F,
//          double tol,
//          double timelimit,
//          int maxiter,
//          World & dw) ;

/**
 * \brief ALS method for CP decomposition with dimension tree PP subroutine
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_DT_sub(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, Matrix<> *dW,
                    Matrix<> *F, double tol, double tol_init, double timelimit,
                    int maxiter, double &st_time, ofstream &Plot_File,
                    double &projnorm, int &iter, int resprint, World &dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_PP_sub(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, Matrix<> *dW,
                    Matrix<> *F, double tol, double tol_init, double timelimit,
                    int maxiter, double &st_time, double lambda,
                    double ratio_step, ofstream &Plot_File, double &projnorm,
                    int &iter, int resprint, bool bench, World &dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_PP_partupdate_sub(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W,
                               Matrix<> *dW, Matrix<> *F, double tol,
                               double tol_init, double timelimit, int maxiter,
                               double update_percentage, double &st_time,
                               double lambda, double ratio_step,
                               ofstream &Plot_File, double &projnorm, int &iter,
                               int resprint, bool bench, World &dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_PP(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, Matrix<> *F,
              double tol, double tol_init, double timelimit, int maxiter,
              double lambda, double ratio_step, ofstream &Plot_File,
              int resprint, bool bench, World &dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_PP_partupdate(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W,
                         Matrix<> *F, double tol, double tol_init,
                         double timelimit, int maxiter, double lambda,
                         double ratio_step, double update_percentage,
                         ofstream &Plot_File, int resprint, bool bench,
                         World &dw);

vector<int> sort_indexes(const vector<double> &v);

#endif
