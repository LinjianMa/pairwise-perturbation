#ifndef __ALS_CP_H__
#define __ALS_CP_H__

#include <ctf.hpp>
#include <fstream>
#include <tuple>
#include <unordered_map>
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
			  double lambda,
        	  ofstream & Plot_File,
        	  int resprint,
        	  bool bench,
			  World & dw) ;

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

void build_V(Tensor<> & V,
			 Matrix<> * W,
			 int order,
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
double alsCP_DT_sub(Tensor<> & V,
					  Matrix<> * W,
        	  		  Matrix<> * grad_W,
					  Matrix<> * dW,
					  Matrix<> * F,
					  double tol,
					  double tol_init,
					  double timelimit,
					  int maxiter,
					  double & st_time,
					  ofstream & Plot_File,
					  double & projnorm,
					  int & iter,
					  int resprint,
					  World & dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_PP_sub(Tensor<> & V,
				  	  Matrix<> * W,
        	  		  Matrix<> * grad_W,
				  	  Matrix<> * dW,
				  	  Matrix<> * F,
				  	  double tol,
				  	  double tol_init,
				  	  double timelimit,
				  	  int maxiter,
				  	  double & st_time,
				  	  ofstream & Plot_File,
				  	  double & projnorm,
				  	  int & iter,
				  	  int resprint,
				  	  bool bench,
				  	  World & dw);

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_PP(Tensor<> & V,
        	  Matrix<> * W,
        	  Matrix<> * grad_W,
        	  Matrix<> * F,
        	  double tol,
        	  double tol_init,
        	  double timelimit,
        	  int maxiter,
        	  double lambda,
        	  double ratio_step,
          	  ofstream & Plot_File,
          	  int resprint,
          	  bool bench,
          	  World & dw) ;


bool alsCP_rank1(Tensor<> & V,
		   Matrix<> * W,
		   Matrix<> * grad_W,
		   double tol,
       double tol_rank1,
		   double timelimit,
		   int maxiter,
		   World & dw);


void build_BDT(unordered_map<string, string> &parent_map, char* seq, int start, int end);

void build_1st_level(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor1, Tensor<> *cached_tensor2, World &dw);

void build_left_child(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor2, World &dw);

void build_right_child(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor1, World &dw);

void update_cached_tensor(Tensor<> &V, Matrix<> *W, Tensor<>* cached_tensor, char* seq, int i);

void fill_mttkrp_tree(unordered_map<string, Tensor<>>mttkrp_map, Matrix<> *W, char *seq, int start, int end, World &dw);

void fill_gamma_tree(unordered_map<string, Matrix<>> &gamma_map, Matrix<> *S, char* seq, int start, int end);

void update_gamma_tree(unordered_map<string, Matrix<>> &gamma_map, Matrix<>* S, char* seq, int index, int start, int end);

tuple<int, int> find_interval(int index, int start, int end);

void compute_gamma(Matrix<> &res, Matrix<> *W, int index, int start, int end);

void compute_M(Matrix<> &M, Tensor<> &res, Matrix<> *W, int index, int start, int end, World &dw);




#endif
