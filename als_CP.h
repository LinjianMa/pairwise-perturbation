#ifndef __ALS_CP_H__
#define __ALS_CP_H__

#include <ctf.hpp>
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
bool alsCP(Tensor<> & V, 
       Matrix<> * W, 
       Matrix<> * grad_W, 
       Matrix<> * F,
       double tol, 
       double timelimit, 
       int maxiter, 
       World & dw);

void mttkrp_map_DT(map<string,Tensor<>>& mttkrp_map, 
           map<string,string>& parent, 
           map<string,string>& sibling, 
           Tensor<>& V, 
           Matrix<> * W, 
           string args,
           World& dw) ;

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
bool alsCP_DT(Tensor<> & V, 
        Matrix<> * W, 
        Matrix<> * grad_W, 
        Matrix<> * F,
        double tol, 
        double timelimit, 
        int maxiter, 
        bool PP,    // whether it works as preconditioning
        World & dw);

// [cd] --> [ab*]
void stringbuilder_mttkrp(char* seq, 
           char* seq_return,
           int N, 
           World & dw);

void Build_mttkrp_map(map<string, Tensor<>> & mttkrp_map, 
            Tensor<> & V, 
            Matrix<> * W,
            char* seq,
            World & dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *  F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_mod(Tensor<> & V, 
         Matrix<> * W, 
         Matrix<> * grad_W, 
         Matrix<> * F,
         double tol, 
         double timelimit, 
         int maxiter, 
         World & dw) ;


#endif