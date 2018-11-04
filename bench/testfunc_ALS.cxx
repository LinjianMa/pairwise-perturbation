/** \addtogroup examples 
  * @{ 
  * \defgroup TESTS_multigrid TESTS_multigrid
  * @{ 
  * \brief NTF/TF multigrid tests
  */
#include "als_Tucker.h"
#include "als_CP.h"
#include "common.h"
#include "testfunc_ALS.h"
//#define ERR_REPORT

#ifndef TEST_SUITE

// /**
//  * \brief CP decomposition of dense tensor using simple als
//  *        This test is not functional now
//  */
// void TEST_dense_uniform_alsCP(int s, 
// 							  int K, 
// 							  World & dw){

// 	double st_time = MPI_Wtime();

// 	int * lens = new int[3];
// 	for (int i=0; i<3; i++) lens[i]=s;
// 	Tensor<>* V = new Tensor<>(3, lens, dw); 

// 	// build V tensor
// 	int64_t my_tot_nnz = s*s*s;
// 	int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
// 	double * vals = (double*)malloc(sizeof(double)*my_tot_nnz);
// 	for (int64_t i=0; i<s; i++)
// 	for (int64_t j=0; j<s; j++)
// 	for (int64_t k=0; k<s; k++) {
// 		inds[i+j*s+k*s*s] = i+j*s+k*s*s;
// 		if (dw.rank==0) vals[i+j*s+k*s*s] = pow((i+1)*(i+1)+(j+1)*(j+1)+(k+1)*(k+1),-0.5);
// 		else vals[i+j*s+k*s*s] = 0.;
// 	}
// 	V->write(my_tot_nnz, inds, vals);
// 	free(inds);
// 	free(vals);
	
// 	//if (dw.rank==0) cout << "V_origin" << endl;
// 	//(*V).print();

// 	Matrix<>* W = new Matrix<>[3];
// 	Matrix<>* grad_W = new Matrix<>[3];
// 	for (int i=0; i<3; i++) {
// 		W[i] = Matrix<>(V->lens[i],K);
// 		grad_W[i] = Matrix<>(V->lens[i],K);
// 		W[i].fill_random(0,0.1);  //work for n=4
// 		grad_W[i].fill_random(0,1);  
// 	}
// 	//construct F matrices
// 	Matrix<>* F = new Matrix<>[3];
// 	for (int i=0; i<3; i++) {
// 		F[i] = Matrix<>(V->lens[i],K);
// 		F[i]["ij"] = 0.;
// 	}	
// 	int relaxiter[3]; for (int i=0; i<3; i++) relaxiter[i]=50;
// 	//initnorm
// 	gradient_CP(*V, W, grad_W, dw);
// 	//double initnorm = 0;
// 	//for (int i=0; i<V->order; i++) { 
// 	//	initnorm += grad_W[i].norm2();
// 	//}
// 	double initnorm = V->norm2();
// 	if(dw.rank==0) printf("Init gradient norm %E \n",initnorm);
// 	//bool
// 	bool finished = false;
// 	while (finished==false) {
// 		//finished = alstf(*V, W, grad_W, F, 1e-10*initnorm, 5000, 5000, dw);
// 		//finished = CP_AMG_mult(*V, W, grad_W, F, 0.0000001*initnorm, 200, 300, relaxiter, "alstf", dw);
// 		//finished = CP_FAS(*V, W, grad_W, F, 0.000001*initnorm, 500, 500, relaxiter, "alstf", dw);
// 	}
	
// 	//Tensor<double>* V_out = new Tensor<double>(3, lens, dw); 
// 	//(*V_out)["ijl"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"];
// 	//if (dw.rank==0) cout << "V comparison" << endl;
// 	//(*V).print();
// 	//(*V_out).print();

// 	printf ("TEST_dense_uniform_alstf took %lf seconds\n",MPI_Wtime()-st_time);
// } 


/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_collinearity_CP(int N,				// Dimension of the tensor
						int s,				// size in each dimension
						int K, 			// Decomposition rank
						bool sparse_V,		// Whether V is set to be sparse or not
						double tol_grad,
						double tol_init,
						double lambda,
						double ratio_step,
						double col_min, 
						double col_max,
						double ratio_noise,
        				ofstream & Plot_File,
						World & dw){
	if(dw.rank==0) printf("Test of tensor made by matrices with collinearity constraint\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	for (int i=0; i<N; i++) lens[i]=s;
	char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
	char arg[N+1];
	arg[N] = '\0';
	for (int i = 0; i < N; i++) {
		arg[i] = chars[i];
	}

	Tensor<> V = Gen_collinearity(lens, N, K, col_min, col_max, dw); 
	Tensor<> V_noise = Tensor<>(N, sparse_V, lens, dw);
	V_noise.fill_random(-1,1);
	double noise_norm = V_noise.norm2();
	double V_norm = V.norm2();
	V_noise[arg] = ratio_noise*V_norm/noise_norm*V_noise[arg];
	V[arg] = V[arg] + V_noise[arg];

	Matrix<>* W = new Matrix<>[N];				// N matrices V will be decomposed into
	Matrix<>* grad_W = new Matrix<>[N];			// gradients in N dimensions 
	for (int i=0; i<N; i++) {
		W[i] = Matrix<>(s,K,dw);
		grad_W[i] = Matrix<>(s,K,dw);
		W[i].fill_random(0,1); 
		grad_W[i].fill_random(0,1);  
	}

	//construct F matrices (correction terms, F[]=0 initially)
	Matrix<>* F = new Matrix<>[N];
	for (int i=0; i<N; i++) {
		F[i] = Matrix<>(V.lens[i],K,dw);
		F[i]["ij"] = 0.;
	}	
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("Norm of V %E \n",Vnorm);
	bool finished = false;
	// Run ALS
	// while (finished == false) {
		// finished = alsCP(V, W, grad_W, F, tol_grad*Vnorm, 400, 400, dw);
		// finished = alsCP_DT(V, W, grad_W, F, tol_grad*Vnorm, 80000, 80000, lambda, Plot_File, dw);
		// finished = alsCP_mod(V, W, grad_W, F, tol_grad*Vnorm, 4000000, 4000000, dw);
		// finished = alsCP_PP(V, W, grad_W, F, tol_grad*Vnorm, tol_init, 200000, 200000, lambda, ratio_step, Plot_File, dw);
	//}
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_randmat_CP(int N,				// Dimension of the tensor
					 int s,				// size in each dimension
					 int K, 			// Decomposition rank
					 bool sparse_V,		// Whether V is set to be sparse or not
					 double tol_grad,
					 double tol_init,
					 double lambda,
					 double ratio_step,
        			 ofstream & Plot_File,
					 World & dw){
	if(dw.rank==0) printf("Test of tensor made by random matrices\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	for (int i=0; i<N; i++) lens[i]=s;
	Tensor<> V; 
	Matrix<>* W = new Matrix<>[N];				// N matrices V will be decomposed into
	Matrix<>* grad_W = new Matrix<>[N];			// gradients in N dimensions 
	for (int i=0; i<N; i++) {
		W[i] = Matrix<>(s,K,dw);
		grad_W[i] = Matrix<>(s,K,dw);
		W[i].fill_random(-1,1); 
		grad_W[i].fill_random(0,1);  
	}
	// if (dw.rank==0) cout << "here" << endl;
	build_V(V, W, N, dw);
	// if (dw.rank==0) cout << "there" << endl;
	for (int i=0; i<N; i++) {
		W[i].fill_random(-1,1); 
	}
	//construct F matrices (correction terms, F[]=0 initially)
	Matrix<>* F = new Matrix<>[N];
	for (int i=0; i<N; i++) {
		F[i] = Matrix<>(V.lens[i],K,dw);
		F[i]["ij"] = 0.;
	}	
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("Norm of V %E \n",Vnorm);
	bool finished = false;
	// Run ALS
	// while (finished == false) {
	// 	// finished = alsCP(V, W, grad_W, F, tol_grad*Vnorm, 400, 400, dw);
	// 	// finished = alsCP_DT(V, W, grad_W, F, tol_grad*Vnorm, 80000, 80000, lambda, Plot_File, dw);
	// 	// finished = alsCP_mod(V, W, grad_W, F, tol_grad*Vnorm, 4000000, 4000000, dw);
	// 	finished = alsCP_PP(V, W, grad_W, F, tol_grad*Vnorm, tol_init, 200000, 200000, lambda, ratio_step, Plot_File, dw);
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_3d_poisson_CP(int N,				// Dimension of the tensor
					 int s,				// size in each dimension
					 int K, 			// Decomposition rank
					 bool sparse_V,		// Whether V is set to be sparse or not
					 double tol_grad,
					 double tol_init,
					 double lambda,
					 double ratio_step,
        			 ofstream & Plot_File,
					 World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian function\n");
	double st_time = MPI_Wtime();
	int * lens0 = new int[N];
	for (int i=0; i<N; i++) lens0[i]=s;
	Tensor<>* V0 = new Tensor<>(N, sparse_V, lens0, dw); 
	laplacian_tensor(*V0, N, s, sparse_V, dw);
	// reshape V0
	int * lens = new int[N/2];
	for (int i=0; i<N/2; i++) lens[i]=s*s;
	Tensor<>* V = new Tensor<>(N/2, sparse_V, lens, dw); 
	// reshape V0 into V
	fold_unfold(*V0, *V);
	//V0->print();
	//V->print();
	N = N/2;

	Matrix<>* W = new Matrix<>[N];				// N matrices V will be decomposed into
	Matrix<>* grad_W = new Matrix<>[N];			// gradients in N dimensions 
	for (int i=0; i<N; i++) {
		W[i] = Matrix<>(V->lens[i],K,dw);
		grad_W[i] = Matrix<>(V->lens[i],K,dw);
		W[i].fill_random(0,1); 
		grad_W[i].fill_random(0,1);  
	}
	//construct F matrices (correction terms, F[]=0 initially)
	Matrix<>* F = new Matrix<>[N];
	for (int i=0; i<N; i++) {
		F[i] = Matrix<>(V->lens[i],K,dw);
		F[i]["ij"] = 0.;
	}	
	// Norm of V
	double Vnorm = V->norm2();
	if(dw.rank==0) printf("Norm of V %E \n",Vnorm);
	// bool finished = false;
	// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	Tensor<> residule(N, lens, dw);
	// Run ALS
	// while (finished == false) {
	// 	// finished = alsCP(*V, W, grad_W, F, 1e-10*Vnorm, 400, 400, dw);
	// 	// finished = alsCP_DT(*V, W, grad_W, F, 1e-10*Vnorm, 80000, 80000, lambda, Plot_File, dw);
	// 	// finished = alsCP_mod(*V, W, grad_W, F, 1e-10*Vnorm, 4000000, 4000000, dw);
	// 	finished = alsCP_PP(*V, W, grad_W, F, tol_grad*Vnorm, tol_init, 200000, 200000, lambda, ratio_step, Plot_File, dw);
	// 	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	// 	residule["ijlm"] = (*V2)["ijlm"]-(*V)["ijlm"];
	// 	double norm = residule.norm2();
	// 	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_poisson_CP(int N,				// Dimension of the tensor
					 int s,				// size in each dimension
					 int K, 			// Decomposition rank
					 bool sparse_V,		// Whether V is set to be sparse or not
					 double tol_init,
					 double lambda,
					 double ratio_step,
        			 ofstream & Plot_File,
					 World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian function\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	for (int i=0; i<N; i++) lens[i]=s;
	Tensor<>* V = new Tensor<>(N, sparse_V, lens, dw); 
	laplacian_tensor(*V, N, s, sparse_V, dw);

	Matrix<>* W = new Matrix<>[N];				// N matrices V will be decomposed into
	Matrix<>* grad_W = new Matrix<>[N];			// gradients in N dimensions 
	for (int i=0; i<N; i++) {
		W[i] = Matrix<>(V->lens[i],K,dw);
		grad_W[i] = Matrix<>(V->lens[i],K,dw);
		W[i].fill_random(0,1); 
		grad_W[i].fill_random(0,1);  
	}
	//construct F matrices (correction terms, F[]=0 initially)
	Matrix<>* F = new Matrix<>[N];
	for (int i=0; i<N; i++) {
		F[i] = Matrix<>(V->lens[i],K,dw);
		F[i]["ij"] = 0.;
	}	
	// Norm of V
	double Vnorm = V->norm2();
	if(dw.rank==0) printf("Norm of V %E \n",Vnorm);
	// bool finished = false;
	// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	// Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	Tensor<> residule(N, lens, dw);
	// Run ALS
	// while (finished == false) {
	// 	// finished = alsCP(*V, W, grad_W, F, 1e-10*Vnorm, 400, 400, dw);
	// 	// finished = alsCP_DT(*V, W, grad_W, F, 1e-10*Vnorm, 80000, 80000, lambda, Plot_File, dw);
	// 	// finished = alsCP_mod(*V, W, grad_W, F, 1e-10*Vnorm, 4000000, 4000000, dw);
	// 	finished = alsCP_PP(*V, W, grad_W, F, 1e-10*Vnorm, tol_init, 200000, 200000, lambda, ratio_step, Plot_File, dw);
	// 	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	// 	residule["ijlm"] = (*V2)["ijlm"]-(*V)["ijlm"];
	// 	double norm = residule.norm2();
	// 	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief test the Hosvd method
 */
void TEST_alsTucker(int N,
					int * V_lens,
					int * ranks,
					World & dw){
	if(dw.rank==0) printf("Test of Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	Tensor<> V(N, V_lens, dw);
	V.fill_random(0,1);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	bool finished = false;
	while (finished == false) {
		finished = alsTucker(V, hosvd_core, W, 1e-10*Vnorm, 40000, 40000, dw);
		// check the residule
		Matrix<>* W_T = new Matrix<>[N];
		for (int i=0; i<N; i++) {
			W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
			W_T[i]["ij"] = W[i]["ji"];
		}
		Tensor<> V_check(N, V_lens, dw);
		Tensor<> V_diff(N, V_lens, dw);
		TTMc(V_check, hosvd_core, W_T, -1, dw);
		char seq[V.order+1];
		seq[V.order] = '\0';
		for (int jj=0; jj<V.order; jj++) {
			seq[jj] = 'a'+jj;
		}
		V_diff[seq] = V_check[seq] - V[seq];
		double diffnorm_V = V_diff.norm2();
		if(dw.rank==0) printf("diff Norm of V =%lf\n", diffnorm_V); 
	}
	if(dw.rank==0) printf ("TEST_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief test the Hosvd method
 */
void TEST_alsTucker_DT(int N,
					   int * V_lens,
					   int * ranks,
					   World & dw){
	if(dw.rank==0) printf("Test of Tucker Decomposition with dimension tree\n");
	double st_time = MPI_Wtime();
	Tensor<> V(N, V_lens, dw);
	V.fill_random(0,1);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	bool finished = false;
	while (finished == false) {
		// finished = alsTucker_DT(V, hosvd_core, W, 1e-10*Vnorm, 1000000, 100000, dw);
 	// 	// check the residule
		// Matrix<>* W_T = new Matrix<>[N];
		// for (int i=0; i<N; i++) {
		// 	W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
		// 	W_T[i]["ij"] = W[i]["ji"];
		// }
		// Tensor<> V_check(N, lens, dw);
		// Tensor<> V_diff(N, lens, dw);
		// TTMc(V_check, hosvd_core, W_T, -1, dw);
		// char seq[V.order+1];
		// seq[V.order] = '\0';
		// for (int jj=0; jj<V.order; jj++) {
		// 	seq[jj] = 'a'+jj;
		// }
		// V_diff[seq] = V_check[seq] - V[seq];
		// double diffnorm_V = V_diff.norm2();
		// if(dw.rank==0) printf("diff Norm of V =%lf\n", diffnorm_V); 
	}
	if(dw.rank==0) printf ("TEST_alsTucker_DT took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker(int N,				// Dimension of the tensor
									 int s,				// size in each dimension
									 int K, 			// Decomposition rank
									 bool sparse_V,		// Whether V is set to be sparse or not
									 double criteria,    // global stopping criteria
									 ofstream & Plot_File, 
									 World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	laplacian_tensor(V, N, s, sparse_V, dw);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker_PP(int N,				// Dimension of the tensor
										int s,				// size in each dimension
										int K, 			// Decomposition rank
										bool sparse_V,		// Whether V is set to be sparse or not
										double criteria,    // global stopping criteria
										double tol_init,
										ofstream & Plot_File,
										World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	laplacian_tensor(V, N, s, sparse_V, dw);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);

	// alsTucker_PP(V, hosvd_core, W, criteria*Vnorm, tol_init, 200000000, 200000, Plot_File, dw);

	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_laplacian_alsTucker(int N,				// Dimension of the tensor
									 int s,				// size in each dimension
									 int K, 			// Decomposition rank
									 bool sparse_V,		// Whether V is set to be sparse or not
									 double criteria,    // global stopping criteria
									 ofstream & Plot_File, 
									 World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	random_laplacian_tensor(V, N, s, sparse_V, dw);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_laplacian_alsTucker_PP(int N,				// Dimension of the tensor
										int s,				// size in each dimension
										int K, 			// Decomposition rank
										bool sparse_V,		// Whether V is set to be sparse or not
										double criteria,    // global stopping criteria
										double tol_init,
										ofstream & Plot_File,
										World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	random_laplacian_tensor(V, N, s, sparse_V, dw);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);

	// alsTucker_PP(V, hosvd_core, W, criteria*Vnorm, tol_init, 200000000, 200000, Plot_File, dw);

	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_alsTucker(int N,				// Dimension of the tensor
									 int s,				// size in each dimension
									 int K, 			// Decomposition rank
									 bool sparse_V,		// Whether V is set to be sparse or not
									 double criteria,    // global stopping criteria
									 ofstream & Plot_File, 
									 World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	V.fill_random(-1,1);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_alsTucker_PP(int N,				// Dimension of the tensor
										int s,				// size in each dimension
										int K, 			// Decomposition rank
										bool sparse_V,		// Whether V is set to be sparse or not
										double criteria,    // global stopping criteria
										double tol_init,
										ofstream & Plot_File,
										World & dw){
	if(dw.rank==0) printf("Test of sparse laplacian Tucker Decomposition\n");
	double st_time = MPI_Wtime();
	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N, sparse_V, lens, dw); 
	V.fill_random(-1,1);
	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);

	// alsTucker_PP(V, hosvd_core, W, criteria*Vnorm, tol_init, 200000000, 200000, Plot_File, dw);

	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}  

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsTucker(int s, 
							  	  int K, 
								  bool sparse_V,		// Whether V is set to be sparse or not
								  double criteria,    // global stopping criteria
								  ofstream & Plot_File,
								  World & dw){
	int N=6;

	double st_time = MPI_Wtime();

	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<> V = Tensor<>(N, lens, dw); 

	// build V tensor
	int64_t my_tot_nnz = s*s*s*s*s*s;
	int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
	double * vals = (double*)malloc(sizeof(double)*my_tot_nnz);
	for (int64_t i=0; i<s; i++)
	for (int64_t j=0; j<s; j++)
	for (int64_t k=0; k<s; k++) 
	for (int64_t l=0; l<s; l++)
	for (int64_t m=0; m<s; m++)
	for (int64_t n=0; n<s; n++){
		inds[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s;
		if (dw.rank==0) vals[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = pow((i+1)*(i+1)+(j+1)*(j+1)+(k+1)*(k+1)+(l+1)*(l+1)+(m+1)*(m+1)+(n+1)*(n+1),-0.5);
		else vals[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = 0.;
	}
	V.write(my_tot_nnz, inds, vals);
	free(inds);
	free(vals);

	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }

	if (dw.rank==0)printf ("TEST_dense_uniform_alsTucker took %lf seconds\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsTucker_PP(int s, 
							  		 int K, 
									 bool sparse_V,		// Whether V is set to be sparse or not
									 double criteria,    // global stopping criteria
									 double tol_init,
									 ofstream & Plot_File,
									 World & dw){
	int N=6;

	double st_time = MPI_Wtime();

	int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		lens[i] = s;
		ranks[i] = K;
	}
	Tensor<> V = Tensor<>(N, lens, dw); 

	// build V tensor
	int64_t my_tot_nnz = s*s*s*s*s*s;
	int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
	double * vals = (double*)malloc(sizeof(double)*my_tot_nnz);
	for (int64_t i=0; i<s; i++)
	for (int64_t j=0; j<s; j++)
	for (int64_t k=0; k<s; k++) 
	for (int64_t l=0; l<s; l++)
	for (int64_t m=0; m<s; m++)
	for (int64_t n=0; n<s; n++){
		inds[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s;
		if (dw.rank==0) vals[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = pow((i+1)*(i+1)+(j+1)*(j+1)+(k+1)*(k+1)+(l+1)*(l+1)+(m+1)*(m+1)+(n+1)*(n+1),-0.5);
		else vals[i+j*s+k*s*s+l*s*s*s+m*s*s*s*s+n*s*s*s*s*s] = 0.;
	}
	V.write(my_tot_nnz, inds, vals);
	free(inds);
	free(vals);

	// Norm of V
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);

	// alsTucker_PP(V, hosvd_core, W, criteria*Vnorm, tol_init, 200000000, 200000, Plot_File, dw);

	if (dw.rank==0)printf ("TEST_dense_uniform_alsTucker took %lf seconds\n",MPI_Wtime()-st_time);
} 

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_3d_poisson_Tucker(int N,				// Dimension of the tensor
							int s,				// size in each dimension
							int K, 			// Decomposition rank
							bool sparse_V,		// Whether V is set to be sparse or not
							double criteria,
							ofstream & Plot_File,
							World & dw){
	if(dw.rank==0) printf("Test of 3d possion Tucker decomposition\n");
	double st_time = MPI_Wtime();
	int * lens0 = new int[N];
	for (int i=0; i<N; i++) lens0[i]=s;
	Tensor<>V0 = Tensor<>(N, sparse_V, lens0, dw); 
	laplacian_tensor(V0, N, s, sparse_V, dw);
	// reshape V0
	int * lens = new int[N/2];
	int * ranks = new int[N/2];
	for (int i=0; i<N/2; i++) {
		lens[i]=s*s;
		ranks[i] = K;
	}
	Tensor<>V = Tensor<>(N/2, sparse_V, lens, dw); 
	Tensor<>V_rand = Tensor<>(N/2, sparse_V, lens, dw); 
	V_rand.fill_random(0,0.3);

	// reshape V0 into V
	fold_unfold(V0, V);
	V["abcd"] += V_rand["abcd"];
	//V0->print();
	// V.print();
	N = N/2;
	double Vnorm = V.norm2();
	
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }

	if (dw.rank==0)printf ("TEST_dense_uniform_alsTucker took %lf seconds\n",MPI_Wtime()-st_time);
} 

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_construct_Tucker(int N,				// Dimension of the tensor
						   int s,				// size in each dimension
						   int K, 			// Decomposition rank
						   bool sparse_V,		// Whether V is set to be sparse or not
						   double criteria,
						   ofstream & Plot_File,
						   World & dw){
	if(dw.rank==0) printf("Test of construct Tucker decomposition\n");
	double st_time = MPI_Wtime();
	int * lens0 = new int[N];
	for (int i=0; i<N; i++) lens0[i]=s;
	Tensor<>V = Tensor<>(N, sparse_V, lens0, dw); 
	Matrix<>* W = new Matrix<>[N];

	int lens[] = {s,s,s,s,s,s};
	Vector<>** r1 =  Gen_vector_condition(lens, N, 3, 1);
	for (int i=0; i<3; i++)
	for (int j=0; j<3; j++)
	for (int k=0; k<3; k++)
	for (int l=0; l<3; l++)
	for (int a=0; a<3; a++)
	for (int b=0; b<3; b++) {
		double cv = double(rand() % 1000)/100;
		if (dw.rank==0) cout << cv << endl;
		V["abcdef"] += cv*r1[0][i]["a"]*r1[1][j]["b"]*r1[2][k]["c"]*r1[3][l]["d"]*r1[4][l]["e"]*r1[5][l]["f"];
	}

	// int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		// lens[i] = s;
		ranks[i] = K;
	}

	Tensor<>V_rand = Tensor<>(N, sparse_V, lens, dw); 
	// V_rand.fill_random(-0.3,0.3);

	// reshape V0 into V
	// fold_unfold(V0, V);
	V["abcdef"] += V_rand["abcdef"];

	// Matrix<>* W0 = new Matrix<>[N];
	// for (int i=0; i<N; i++ ) {
	// 	W0[i] = Matrix<>(7,s,dw);
	// 	W0[i].fill_random(0,1);
	// }
	// int * lens1 = new int[N];
	// for (int i=0; i<N; i++) lens1[i]=7;	
	// Tensor<> core = Tensor<>(N, lens1, dw);
	// core.fill_random(0,1);
	// TTMc(V, core, W0, -1, dw);
	// // V["abcdef"] = core["hijklm"]*W0[0]["ah"]*W0[1]["bi"]*W0[2]["cj"]*W0[3]["dk"]*W0[4]["el"]*W0[5]["fm"];




	//V0->print();
	// V.print();
	double Vnorm = V.norm2();
	
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	// bool finished = false;
	// while (finished == false) {
	// 	finished = alsTucker_DT(V, hosvd_core, W, criteria*Vnorm, 100000000, 10000, Plot_File, dw);
	// }

	if (dw.rank==0)printf ("TEST_dense_uniform_alsTucker took %lf seconds\n",MPI_Wtime()-st_time);
} 

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_construct_Tucker_pp(int N,				// Dimension of the tensor
						   int s,				// size in each dimension
						   int K, 			// Decomposition rank
						   bool sparse_V,		// Whether V is set to be sparse or not
						   double criteria,
						   double tol_init,
						   ofstream & Plot_File,
						   World & dw){
	if(dw.rank==0) printf("Test of construct Tucker decomposition\n");
	double st_time = MPI_Wtime();
	int * lens0 = new int[N];
	for (int i=0; i<N; i++) lens0[i]=s;
	Tensor<>V = Tensor<>(N, sparse_V, lens0, dw); 
	Matrix<>* W = new Matrix<>[N];

	int lens[] = {s,s,s,s,s,s};
	Vector<>** r1 =  Gen_vector_condition(lens, N, 2, 1);
	for (int i=0; i<2; i++)
	for (int j=0; j<2; j++)
	for (int k=0; k<2; k++)
	for (int l=0; l<2; l++)
	for (int a=0; a<2; a++)
	for (int b=0; b<2; b++) {
		double cv = double(rand() % 1000)/10+5;
		V["abcdef"] += cv*r1[0][i]["a"]*r1[1][j]["b"]*r1[2][k]["c"]*r1[3][l]["d"]*r1[4][l]["e"]*r1[5][l]["f"];
	}

	// int * lens = new int[N];
	int * ranks = new int[N];
	for (int i=0; i<N; i++) {
		// lens[i] = s;
		ranks[i] = K;
	}

	Tensor<>V_rand = Tensor<>(N, sparse_V, lens, dw); 
	V_rand.fill_random(-0.3,0.3);

	// reshape V0 into V
	// fold_unfold(V0, V);
	V["abcdef"] += V_rand["abcdef"];
	//V0->print();
	// V.print();
	double Vnorm = V.norm2();
	
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);

	// alsTucker_PP(V, hosvd_core, W, criteria*Vnorm, tol_init, 200000000, 200000, Plot_File, dw);

	if (dw.rank==0)printf ("TEST_dense_uniform_alsTucker took %lf seconds\n",MPI_Wtime()-st_time);
} 

#endif
