#ifndef __CP_H__
#define __CP_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template<typename dtype>  
class CPD : public Decomposition<dtype> {

	public:

		// grad_W: gradient in each dimension
		Matrix<> * grad_W = NULL;
		double gradnorm = 0.;

		CPD(int order, int size, int r, World & dw);

		CPD(int order, int* size, int* r, World & dw);

		void Init(Tensor<dtype>* input, Matrix<dtype>* mat);

		~CPD();

		void print_grad(int i) const;

		void update_gradnorm();


		/**
		 * \brief ALS method for CP decomposition
		 *  tol: tolerance for a relative stopping condition
		 *  timelimit, maxiter: limit of time and iterations
		 */
		bool als(double tol, double timelimit, int maxiter);

		// void mttkrp_map_DT(map<string,Tensor<>>& mttkrp_map, 
		//            map<string,string>& parent, 
		//            map<string,string>& sibling, 
		//            Tensor<>& V, 
		//            Matrix<> * W, 
		//            string args,
		//            World& dw);

		// /**
		//  * \brief ALS method for CP decomposition with decision tree
		//  *  W: output solutions
		//  *  V: input tensor
		//  *  grad_W: gradient in each dimension
		//  *  F: correction terms, F[]=0 initially
		//  *  tol: tolerance for a relative stopping condition
		//  *  timelimit, maxiter: limit of time and iterations
		//  *  V.order should be >=4
		//  */
		// bool alsCP_DT(Tensor<> & V, 
		// 			  Matrix<> * W, 
		// 			  Matrix<> * grad_W, 
		// 			  Matrix<> * F,
		// 			  double tol, 
		// 			  double timelimit, 
		// 			  int maxiter, 
		// 			  double lambda,
		//         	  ofstream & Plot_File,
		//         	  int resprint,
		//         	  bool bench,
		// 			  World & dw) ;

		// // [cd] --> [ab*]
		// void stringbuilder_mttkrp(char* seq, 
		//            char* seq_return,
		//            int N, 
		//            World & dw);

		// void Build_mttkrp_map(map<string, Tensor<>> & mttkrp_map, 
		//             Tensor<> & V, 
		//             Matrix<> * W,
		//             char* seq,
		//             World & dw);

		// void build_V(Tensor<> & V,
		// 			 Matrix<> * W,
		// 			 int order,
		// 			 World & dw);
		// /**
		//  * \brief ALS method for CP decomposition
		//  *  W: output solutions
		//  *  V: input tensor
		//  *  grad_W: gradient in each dimension
		//  *  F: correction terms, F[]=0 initially
		//  *  tol: tolerance for a relative stopping condition
		//  *  timelimit, maxiter: limit of time and iterations
		//  */
		// // bool alsCP_mod(Tensor<> & V, 
		// //          Matrix<> * W, 
		// //          Matrix<> * grad_W, 
		// //          Matrix<> * F,
		// //          double tol, 
		// //          double timelimit, 
		// //          int maxiter, 
		// //          World & dw) ;

		// *
		//  * \brief ALS method for CP decomposition with dimension tree PP subroutine
		//  *  W: output matrices
		//  *  V: input tensor
		//  *  tol: tolerance for a relative stopping condition
		//  *  timelimit, maxiter: limit of time and iterations
		 
		// double alsCP_DT_sub(Tensor<> & V, 
		// 					  Matrix<> * W, 
		//         	  		  Matrix<> * grad_W, 
		// 					  Matrix<> * dW,
		// 					  Matrix<> * F,
		// 					  double tol, 
		// 					  double tol_init,
		// 					  double timelimit, 
		// 					  int maxiter, 
		// 					  double & st_time,
		// 					  ofstream & Plot_File,
		// 					  double & projnorm,
		// 					  int & iter,
		// 					  int resprint,
		// 					  World & dw);

		// /**
		//  * \brief ALS method for CP decomposition
		//  *  W: output matrices
		//  *  V: input tensor
		//  *  tol: tolerance for a relative stopping condition
		//  *  timelimit, maxiter: limit of time and iterations
		//  */
		// double alsCP_PP_sub(Tensor<> & V, 
		// 				  	  Matrix<> * W, 
		//         	  		  Matrix<> * grad_W, 
		// 				  	  Matrix<> * dW,
		// 				  	  Matrix<> * F,
		// 				  	  double tol, 
		// 				  	  double tol_init,
		// 				  	  double timelimit, 
		// 				  	  int maxiter, 
		// 				  	  double & st_time,
		// 				  	  ofstream & Plot_File,
		// 				  	  double & projnorm,
		// 				  	  int & iter,
		// 				  	  int resprint,
		// 				  	  bool bench,
		// 				  	  World & dw);

		// /**
		//  * \brief ALS method for CP decomposition
		//  *  W: output matrices
		//  *  V: input tensor
		//  *  tol: tolerance for a relative stopping condition
		//  *  timelimit, maxiter: limit of time and iterations
		//  */
		// bool alsCP_PP(Tensor<> & V, 
		//         	  Matrix<> * W, 
		//         	  Matrix<> * grad_W, 
		//         	  Matrix<> * F,
		//         	  double tol, 
		//         	  double tol_init,
		//         	  double timelimit, 
		//         	  int maxiter, 
		//         	  double lambda,
		//         	  double ratio_step,
		//           	  ofstream & Plot_File,
		//           	  int resprint,
		//           	  bool bench,
		//           	  World & dw) ;

};

#include "CP.cxx"

#endif
