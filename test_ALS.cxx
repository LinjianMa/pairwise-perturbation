/** \addtogroup examples 
  * @{ 
  * \defgroup TESTS_multigrid TESTS_multigrid
  * @{ 
  * \brief NTF/TF multigrid tests
  */
#include <ctf.hpp>
#include "als_Tucker.cxx"
using namespace CTF;
//#define ERR_REPORT

#ifndef TEST_SUITE

/**
 * \brief CP decomposition of random tensor using simple ALS
 */
void TEST_alsCP(int dim, 
				int * lens, 
				int K, 
				World & dw){
	if(dw.rank==0) printf("Test of alstf function\n");
	Tensor<>* V = new Tensor<>(dim, lens, dw); 
	V->fill_random(10,100);
	if(dw.rank==0) cout << "V_origin" << endl;
	(*V).print();
	Matrix<>* W = new Matrix<>[dim];
	Matrix<>* grad_W = new Matrix<>[dim];
	for (int i=0; i<dim; i++) {
		W[i] = Matrix<>(V->lens[i],K);
		grad_W[i] = Matrix<>(V->lens[i],K);
		W[i].fill_random(0,1);  
		grad_W[i].fill_random(0,1);  
	}
	//construct F matrices
	Matrix<>* F = new Matrix<>[dim];
	for (int i=0; i<dim; i++) {
		F[i] = Matrix<>(V->lens[i],K);
		F[i]["ij"] = 0.;
	}	
	//initnorm
	gradient_CP(*V, W, grad_W, dw);
	double initnorm = V->norm2();
	if(dw.rank==0) printf("Init norm %E \n",initnorm);
	//bool
	bool finished = false;
	while (finished==false) {
		finished = alsCP(*V, W, grad_W, F, 1e-5*initnorm, 5000, 5000, dw);
	}
	Tensor<>* V2 = new Tensor<>(dim, lens, dw); 
	(*V2)["ij"] = W[0]["ik"]*W[1]["jk"];
	Tensor<> residule(dim, lens, dw);
	residule["ij"] = (*V2)["ij"]-(*V)["ij"];
	residule.print();
	double norm = residule.norm2();
	if(dw.rank==0) printf("norm=%lf\n", norm); 
	if(dw.rank==0) printf("alstf test finished\n");
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsCP(int N,				// Dimension of the tensor
								 int s,				// size in each dimension
								 int K, 			// Decomposition rank
								 bool sparse_V,		// Whether V is set to be sparse or not
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
	bool finished = false;
	// Run ALS
	while (finished == false) {
		finished = alsCP(*V, W, grad_W, F, 1e-10*Vnorm, 20000, 20000, dw);
			// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	Tensor<> residule(N, lens, dw);
	residule["ijlk"] = (*V2)["ijlk"]-(*V)["ijlk"];
	double norm = residule.norm2();
	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	}
	// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	Tensor<> residule(N, lens, dw);
	residule["ijlk"] = (*V2)["ijlk"]-(*V)["ijlk"];
	double norm = residule.norm2();
	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 
/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsCP_mod(int N,				// Dimension of the tensor
									 int s,				// size in each dimension
									 int K, 			// Decomposition rank
									 bool sparse_V,		// Whether V is set to be sparse or not
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
	bool finished = false;
	alsCP(*V, W, grad_W, F, 1e-10*Vnorm, 500, 500, dw);
	// Run ALS
	while (finished == false) {
		finished = alsCP_mod(*V, W, grad_W, F, 1e-10*Vnorm, 20000, 20000, dw);
			// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	Tensor<> residule(N, lens, dw);
	residule["ijlk"] = (*V2)["ijlk"]-(*V)["ijlk"];
	double norm = residule.norm2();
	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	}
	// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	(*V2)["ijlm"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"]*W[3]["mk"];
	Tensor<> residule(N, lens, dw);
	residule["ijlk"] = (*V2)["ijlk"]-(*V)["ijlk"];
	double norm = residule.norm2();
	if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsCP(int s, 
							  int K, 
							  World & dw){

	double st_time = MPI_Wtime();

	int * lens = new int[3];
	for (int i=0; i<3; i++) lens[i]=s;
	Tensor<>* V = new Tensor<>(3, lens, dw); 

	// build V tensor
	int64_t my_tot_nnz = s*s*s;
	int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
	double * vals = (double*)malloc(sizeof(double)*my_tot_nnz);
	for (int64_t i=0; i<s; i++)
	for (int64_t j=0; j<s; j++)
	for (int64_t k=0; k<s; k++) {
		inds[i+j*s+k*s*s] = i+j*s+k*s*s;
		if (dw.rank==0) vals[i+j*s+k*s*s] = pow((i+1)*(i+1)+(j+1)*(j+1)+(k+1)*(k+1),-0.5);
		else vals[i+j*s+k*s*s] = 0.;
	}
	V->write(my_tot_nnz, inds, vals);
	free(inds);
	free(vals);
	
	//if (dw.rank==0) cout << "V_origin" << endl;
	//(*V).print();

	Matrix<>* W = new Matrix<>[3];
	Matrix<>* grad_W = new Matrix<>[3];
	for (int i=0; i<3; i++) {
		W[i] = Matrix<>(V->lens[i],K);
		grad_W[i] = Matrix<>(V->lens[i],K);
		W[i].fill_random(0,0.1);  //work for n=4
		grad_W[i].fill_random(0,1);  
	}
	//construct F matrices
	Matrix<>* F = new Matrix<>[3];
	for (int i=0; i<3; i++) {
		F[i] = Matrix<>(V->lens[i],K);
		F[i]["ij"] = 0.;
	}	
	int relaxiter[3]; for (int i=0; i<3; i++) relaxiter[i]=50;
	//initnorm
	gradient_CP(*V, W, grad_W, dw);
	//double initnorm = 0;
	//for (int i=0; i<V->order; i++) { 
	//	initnorm += grad_W[i].norm2();
	//}
	double initnorm = V->norm2();
	if(dw.rank==0) printf("Init gradient norm %E \n",initnorm);
	//bool
	bool finished = false;
	while (finished==false) {
		//finished = alstf(*V, W, grad_W, F, 1e-10*initnorm, 5000, 5000, dw);
		//finished = CP_AMG_mult(*V, W, grad_W, F, 0.0000001*initnorm, 200, 300, relaxiter, "alstf", dw);
		//finished = CP_FAS(*V, W, grad_W, F, 0.000001*initnorm, 500, 500, relaxiter, "alstf", dw);
	}
	
	//Tensor<double>* V_out = new Tensor<double>(3, lens, dw); 
	//(*V_out)["ijl"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"];
	//if (dw.rank==0) cout << "V comparison" << endl;
	//(*V).print();
	//(*V_out).print();

	printf ("TEST_dense_uniform_alstf took %lf seconds\n",MPI_Wtime()-st_time);
} 

/**
 * \brief test the identity tensor
 */
void TEST_identity_tensor(int N, int s, World & dw){
	if(dw.rank==0) printf("Test of identity tensor function\n");
	Tensor<> I2 = identitiy_tensor(N, s, dw);
	I2.print();
	if(dw.rank==0) printf("identity tensor function test finished\n");
}

/**
 * \brief test the svd solver
 */
void TEST_SVD_solve(int N, World & dw){
	if(dw.rank==0) printf("Test of SVD solve function\n");
	Matrix<> W(N,N,dw);
	W.fill_random(0,10);
	Matrix<> S(N,N,dw);
	S.fill_random(0,10);
	Matrix<> M(N,N,dw);
	M["ij"]=W["ik"]*S["kj"];
	Matrix<> W_out(N,N,dw);
	SVD_solve(M, W_out, S);
	Matrix<> residule(N,N,dw);
	residule["ij"] = W_out["ij"]-W["ij"];
	double norm = residule.norm2();
	if(dw.rank==0) printf("norm=%llf\n", norm);
	if(dw.rank==0) printf("SVD function test finished\n");
}

/**
 * \brief test the correctness of Laplacian tensor
 */
void TEST_laplacian_tensor(int N, 
						   int s, 
						   bool sparse_V,
						   World & dw){
	if(dw.rank==0) printf("Test of laplacian tensor builder\n");
	int * lens = new int[N];
	for (int i=0; i<N; i++) lens[i]=s;
	Tensor<>* V = new Tensor<>(N, sparse_V, lens, dw); 
	laplacian_tensor(*V, N, s, sparse_V, dw);
	V->print();
	if(dw.rank==0) printf("laplacian tensor builder test finished\n");
} 

/**
 * \brief test the Gauss Seidel method
 *        Gauss-Seidel relaxation for A*Gamma = F
 */
void TEST_gauss_seidel(int N,
					   int K, 
					   World & dw){
	if(dw.rank==0) printf("Test of gauss seidel function\n");
	Matrix<> A(N,K,dw);
	Matrix<> F(N,K,dw);
	Matrix<> Gamma(K,K,dw);
	F.fill_random(0,1);
	Gamma["ij"] = F["ki"]*F["kj"];
	Gauss_Seidel(A, F, Gamma, 100);
	Matrix<> F_out(N,K,dw);
	F_out["ij"] = A["ik"]*Gamma["kj"];
	F.print();
	F_out.print();
	// compare M with P*M_even
	if(dw.rank==0) printf("gauss seidel function test finished\n");
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_3d_poisson_CP(int N,				// Dimension of the tensor
					 int s,				// size in each dimension
					 int K, 			// Decomposition rank
					 bool sparse_V,		// Whether V is set to be sparse or not
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
	bool finished = false;
	// Check for the residule of the CP. (Here V2 is hard coded for N=4)
	Tensor<>* V2 = new Tensor<>(N, lens, dw); 
	Tensor<> residule(N, lens, dw);
	// Run ALS
	while (finished == false) {
		finished = alsCP(*V, W, grad_W, F, 1e-10*Vnorm, 200, 200, dw);
		(*V2)["ijl"] = W[0]["ik"]*W[1]["jk"]*W[2]["lk"];
		residule["ijlk"] = (*V2)["ijl"]-(*V)["ijl"];
		double norm = residule.norm2();
		if(dw.rank==0) printf("Residule Norm=%lf\n", norm); 
	}
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alstf took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

/**
 * \brief test the Hosvd method
 */
void TEST_hosvd(int N,
				int * T_lens,
				int * ranks,
				World & dw){
	if(dw.rank==0) printf("Test of hosvd\n");
	Tensor<> T(N, T_lens, dw);
	T.fill_random(0,10);
	if (dw.rank==0) printf("Tensor T \n");
	T.print();
	Matrix<>* hosvd_factor_matrices = new Matrix<>[N];
	Tensor<> hosvd_core;
	hosvd(T, hosvd_core, hosvd_factor_matrices, ranks, dw);
	hosvd_core.print();
	hosvd_factor_matrices[0].print();
	hosvd_factor_matrices[1].print();
	hosvd_factor_matrices[2].print();
	// compare M with P*M_even
	if(dw.rank==0) printf("hosvd test finished\n");
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
void TEST_alsTucker_mod(int N,
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
	alsTucker(V, hosvd_core, W, 1e-10*Vnorm, 100, 100, dw);
	bool finished = false;
	while (finished == false) {
		finished = alsTucker_mod(V, hosvd_core, W, 1e-10*Vnorm, 40000, 40000, dw);
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
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker(int N,				// Dimension of the tensor
									 int s,				// size in each dimension
									 int K, 			// Decomposition rank
									 bool sparse_V,		// Whether V is set to be sparse or not
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
	bool finished = false;
	while (finished == false) {
		finished = alsTucker(V, hosvd_core, W, 1e-10*Vnorm, 20000, 20000, dw);
		// check the residule
		Matrix<>* W_T = new Matrix<>[N];
		for (int i=0; i<N; i++) {
			W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
			W_T[i]["ij"] = W[i]["ji"];
		}
		Tensor<> V_check(N, lens, dw);
		Tensor<> V_diff(N, lens, dw);
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
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker_mod(int N,				// Dimension of the tensor
										 int s,				// size in each dimension
										 int K, 			// Decomposition rank
										 bool sparse_V,		// Whether V is set to be sparse or not
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
	alsTucker(V, hosvd_core, W, 1e-10*Vnorm, 200, 200, dw);
	// Tucker decomposition
	bool finished = false;
	while (finished == false) {
		finished = alsTucker_mod(V, hosvd_core, W, 1e-10*Vnorm, 20000, 20000, dw);
		// check the residule
		Matrix<>* W_T = new Matrix<>[N];
		for (int i=0; i<N; i++) {
			W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
			W_T[i]["ij"] = W[i]["ji"];
		}
		Tensor<> V_check(N, lens, dw);
		Tensor<> V_diff(N, lens, dw);
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
	if(dw.rank==0) printf ("TEST_sparse_laplacian_alsTucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
}  

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_3d_poisson_Tucker(int N,				// Dimension of the tensor
							int s,				// size in each dimension
							int K, 			// Decomposition rank
							bool sparse_V,		// Whether V is set to be sparse or not
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
	// reshape V0 into V
	fold_unfold(V0, V);
	//V0->print();
	//V->print();
	N = N/2;
	double Vnorm = V.norm2();
	if(dw.rank==0) printf("initial Norm of V =%lf\n", Vnorm); 

	Matrix<>* W = new Matrix<>[N];
	Tensor<> hosvd_core;
	// using hosvd to initialize W and hosvd_core
	hosvd(V, hosvd_core, W, ranks, dw);
	// Tucker decomposition
	bool finished = false;
	while (finished == false) {
		finished = alsTucker(V, hosvd_core, W, 1e-10*Vnorm, 20000, 20000, dw);
	}
	// check the residule
	Matrix<>* W_T = new Matrix<>[N];
	for (int i=0; i<N; i++) {
		W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
		W_T[i]["ij"] = W[i]["ji"];
	}
	Tensor<> V_check(N, lens, dw);
	Tensor<> V_diff(N, lens, dw);
	TTMc(V_check, hosvd_core, W_T, -1, dw);
	char seq[V.order+1];
	seq[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
	}
	V_diff[seq] = V_check[seq] - V[seq];
	double diffnorm_V = V_diff.norm2();
	if(dw.rank==0) printf("diff Norm of V =%lf\n", diffnorm_V); 
	if(dw.rank==0) printf ("TEST_3d_poisson_Tucker took %lf seconds\n\n\n",MPI_Wtime()-st_time);
} 

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char ** argv){
	int rank, np;//, n, pass;
	//int const in_num = argc;
	//char ** input_str = argv;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	{
		World dw(argc, argv);

		//srand48(dw.rank*0);

		int lens[2] = {8, 8};
		//TEST_alsCP(2, lens, 8, dw);
		//TEST_sparse_laplacian_alsCP(4, 20, 4, 0, dw); 
		//TEST_sparse_laplacian_alsCP_mod(4, 20, 4, 0, dw); 
		//TEST_dense_uniform_alsCP(100, 5, dw);
		//TEST_3d_poisson_CP(6, 3, 3, 0, dw);

		//TEST_identity_tensor(6, 4, dw);
		//TEST_SVD_solve(6, dw);
		//TEST_laplacian_tensor(4, 8, 1, dw);  // sparse	
		//TEST_gauss_seidel(4, 4, dw);

		int T_lens[] = {10 ,10, 10, 10, 10, 10};
		int ranks[] = {4, 4, 4, 4, 4, 4};
		//TEST_hosvd(3, T_lens, ranks, dw);
		//TEST_alsTucker(6, T_lens, ranks, dw);	
		TEST_alsTucker_mod(6, T_lens, ranks, dw);	
		//TEST_3d_poisson_Tucker(6, 8, 2, 0, dw);
		//TEST_sparse_laplacian_alsTucker(6, 10, 4, 0, dw); 

	}

	MPI_Finalize();
	return 0;
}

#endif
