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


void TEST_Gram_Schmidt() {
	Vector<> A(5);
	Vector<> B(5);
	A.fill_random(0,1);
	B.fill_random(0,1);
	A.print();
	B.print();
	double innerproduct = A["i"]*B["i"];
	Gram_Schmidt(A, B);
	cout << "innerproduct before GS: " << innerproduct << endl;
	A.print();
	B.print();
	innerproduct = A["i"]*B["i"];
	cout << "innerproduc after GS: " << innerproduct << endl;
}

void TEST_Gen_vector_condition(int * lens,
						  		int dim,
						  		int R,
						  		double condition) {
	Vector<>** vec =  Gen_vector_condition(lens, dim, R, condition);
	for (int i=0; i< dim; i++) {
		for (int j=0; j<R; j++) {
			vec[i][j].print();
		}
	}
	double error= 0.;
	for (int i=0; i< dim; i++) {
		for (int j=0; j<R; j++) 
		for (int k=j+1; k<R; k++) {
			double innerproduct = vec[i][j]["i"]*vec[i][k]["i"];
			error += abs(innerproduct);
		}
	}
	cout << "error= " << error << endl;
}