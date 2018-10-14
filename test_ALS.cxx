/** \addtogroup examples 
  * @{ 
  * \defgroup TESTS_multigrid TESTS_multigrid
  * @{ 
  * \brief NTF/TF multigrid tests
  */
#include "testfunc_ALS.cxx"
//#define ERR_REPORT

#ifndef TEST_SUITE

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
	int const in_num = argc;
	char ** input_str = argv; 

	char * model;		// 0 is CP, 1 is Tucker
	char * tensor;		// which tensor    p / p2 / c / r / r2 / o / 
	int pp;				// 0 Dimention tree 1 pairwise perturbation
	/*
	p : poisson operator
	p2 : poisson operator with doubled dimension (decomposition is not accurate)
	c : decomposition of designed tensor with constrained collinearity
	r : decomposition of tensor made by random matrices
	r2 : random tensor
	o : other tensor
	*/
	int dim; 			// number of dimensions
	int s;   			// tensor size in each dimension
	int R;   			// decomposition rank
	int issparse;  	// whether use the sparse routine or not
	double tol;  	// global convergance tolerance
	double pp_res_tol;	// pp restart tolerance 
	double lambda_; 	// regularization param
	double magni;		// pp update magnitude
	char * filename;	// output csv filename
	double col_min;		// collinearity min
	double col_max;		// collinearity max
	double ratio_noise; // collinearity ratio of noise

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	if (getCmdOption(input_str, input_str+in_num, "-model")) {
		model = getCmdOption(input_str, input_str+in_num, "-model");
    	if (model[0] != 'C' && model[0] != 'T') model = "CP";
	} else {
		model = "CP";
	}	
	if (getCmdOption(input_str, input_str+in_num, "-tensor")) {
		tensor = getCmdOption(input_str, input_str+in_num, "-tensor");
	} else {
		tensor = "p";
	}	
	if (getCmdOption(input_str, input_str+in_num, "-pp")) {
		pp = atoi(getCmdOption(input_str, input_str+in_num, "-pp"));
    	if (pp < 0 || pp > 1) pp = 0;
	} else {
		pp = 0;
	}
	if (getCmdOption(input_str, input_str+in_num, "-dim")) {
		dim = atoi(getCmdOption(input_str, input_str+in_num, "-dim"));
    	if (dim < 0) dim = 6;
	} else {
		dim = 6;
	}
	if (getCmdOption(input_str, input_str+in_num, "-size")) {
		s = atoi(getCmdOption(input_str, input_str+in_num, "-size"));
    	if (s < 0) s = 10;
	} else {
		s = 10;
	}
	if (getCmdOption(input_str, input_str+in_num, "-rank")) {
		R = atoi(getCmdOption(input_str, input_str+in_num, "-rank"));
    	if (R < 0 || R > s) R = s/2;
	} else {
		R = s/2;
	}
	if (getCmdOption(input_str, input_str+in_num, "-issparse")) {
		issparse = atoi(getCmdOption(input_str, input_str+in_num, "-issparse"));
    	if (issparse < 0 || issparse > 1) issparse = 0;
	} else {
		issparse = 0;
	}
	if (getCmdOption(input_str, input_str+in_num, "-tol")) {
		tol = atof(getCmdOption(input_str, input_str+in_num, "-tol"));
    	if (tol < 0 || tol > 1) tol = 1e-10;
	} else {
		tol = 1e-10;
	}	
	if (getCmdOption(input_str, input_str+in_num, "-pp_res_tol")) {
		pp_res_tol = atof(getCmdOption(input_str, input_str+in_num, "-pp_res_tol"));
    	if (pp_res_tol < 0 || pp_res_tol > 1) pp_res_tol = 1e-10;
	} else {
		pp_res_tol = 1e-10;
	}
	if (getCmdOption(input_str, input_str+in_num, "-lambda")) {
		lambda_ = atof(getCmdOption(input_str, input_str+in_num, "-lambda"));
    	if (lambda_ < 0 ) lambda_ = 0.;
	} else {
		lambda_ = 0.;
	}	
	if (getCmdOption(input_str, input_str+in_num, "-magni")) {
		magni = atof(getCmdOption(input_str, input_str+in_num, "-magni"));
    	if (magni < 0 ) magni = 1.;
	} else {
		magni = 1.;
	}	
	if (getCmdOption(input_str, input_str+in_num, "-filename")) {
		filename = getCmdOption(input_str, input_str+in_num, "-filename");
	} else {
		filename = "out.csv";
	}	
	if (getCmdOption(input_str, input_str+in_num, "-colmin")) {
		col_min = atof(getCmdOption(input_str, input_str+in_num, "-colmin"));
	} else {
		col_min = 0.5;
	}
	if (getCmdOption(input_str, input_str+in_num, "-colmax")) {
		col_max = atof(getCmdOption(input_str, input_str+in_num, "-colmax"));
	} else {
		col_max = 0.9;
	}	
	if (getCmdOption(input_str, input_str+in_num, "-rationoise")) {
		ratio_noise = atof(getCmdOption(input_str, input_str+in_num, "-rationoise"));
    	if (ratio_noise < 0 ) ratio_noise = 0.01;
	} else {
		ratio_noise = 0.01;
	}	

	{
		World dw(argc, argv);
		srand48(dw.rank*1);

		if (dw.rank==0) {
			cout << "  model=  " << model << "  tensor=  " << tensor << "  pp=  " << pp << endl;
			cout << "  dim=  " << dim << "  size=  " << s << "  rank=  " << R << endl;
			cout << "  issparse=  " << issparse << "  tolerance=  " << tol << "  restarttol=  " << pp_res_tol << endl;
			cout << "  lambda=  " << lambda_ << "  magnitude=  " << magni << "  filename=  " << filename << endl;
			cout << "  col_min=  " << col_min << "  col_max=  " << col_max  << "  rationoise  " << ratio_noise << endl;

		}

		// initialization of tensor
		Tensor<> V;

		if (tensor[0]=='p') {
			if (strlen(tensor)>1 && tensor[1]=='2') {
				//p2 : poisson operator with doubled dimension (decomposition is not accurate)
				int lens[dim];
				for (int i=0; i<dim; i++) lens[i]=s;
				V = Tensor<>(dim, issparse, lens, dw); 
				laplacian_tensor(V, dim, s, issparse, dw);
			}
			else {
				//p : poisson operator
				int lens0[dim];
				for (int i=0; i<dim; i++) lens0[i]=s;
				Tensor<> V0 = Tensor<>(dim, issparse, lens0, dw); 
				laplacian_tensor(V0, dim, s, issparse, dw);
				// reshape V0
				int lens[dim/2];
				for (int i=0; i<dim/2; i++) lens[i]=s*s;
				V = Tensor<>(dim/2, issparse, lens, dw); 
				// reshape V0 into V
				fold_unfold(V0, V);
			}
		}
		else if (tensor[0]=='c') {
			//c : designed tensor with constrained collinearity
			int lens[dim];
			for (int i=0; i<dim; i++) lens[i]=s;
			char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
			char arg[dim+1];
			arg[dim] = '\0';
			for (int i = 0; i < dim; i++) {
				arg[i] = chars[i];
			}
			V = Gen_collinearity(lens, dim, R, col_min, col_max, dw); 
			Tensor<> V_noise = Tensor<>(dim, issparse, lens, dw);
			V_noise.fill_random(-1,1);
			double noise_norm = V_noise.norm2();
			double V_norm = V.norm2();
			V_noise[arg] = ratio_noise*V_norm/noise_norm*V_noise[arg];
			V[arg] = V[arg] + V_noise[arg];
		}
		else if (tensor[0]=='r') {
			if (strlen(tensor)>1 && tensor[1]=='2') {
				//r2 : random tensor
				int lens[dim];
				for (int i=0; i<dim; i++) lens[i]=s;
				V = Tensor<>(dim, issparse, lens, dw); 
				V.fill_random(-1,1);				
			}
			else {
				//r : tensor made by random matrices
				int lens[dim];
				for (int i=0; i<dim; i++) lens[i]=s;
				Matrix<>* W = new Matrix<>[dim];				// N matrices V will be decomposed into
				for (int i=0; i<dim; i++) {
					W[i] = Matrix<>(s,R,dw);
					W[i].fill_random(-0.5,1); 
				}
				build_V(V, W, dim, dw);
			}
		}
		else if (tensor[0]=='o') {
			//o : other tensor
			// TODO
		}

		if (dw.rank==0) cout << "aaaa" << endl;

		double timelimit = 1e5;
		int maxiter = 1e5;
		double Vnorm = V.norm2();
 		ofstream Plot_File(filename); 
		Matrix<>* W = new Matrix<>[dim];				// N matrices V will be decomposed into
		Matrix<>* grad_W = new Matrix<>[dim];			// gradients in N dimensions 
		for (int i=0; i<V.order; i++) {
			W[i] = Matrix<>(V.lens[i],R,dw);
			grad_W[i] = Matrix<>(V.lens[i],R,dw);
			W[i].fill_random(0,1); 
			grad_W[i].fill_random(0,1);  
		}
		if (dw.rank==0) cout << "ccc" << endl;
		//construct F matrices (correction terms, F[]=0 initially)
		Matrix<>* F = new Matrix<>[dim];
		for (int i=0; i<V.order; i++) {
			F[i] = Matrix<>(V.lens[i],R,dw);
			F[i]["ij"] = 0.;
		}	
		if (dw.rank==0) cout << "bbbb" << endl;

		if (model[0]=='C') {
			if (pp==0) {
				alsCP_DT(V, W, grad_W, F, tol*Vnorm, timelimit, maxiter, lambda_, Plot_File, dw);
			}
			else if (pp==1) {
				alsCP_PP(V, W, grad_W, F, tol*Vnorm, pp_res_tol, timelimit, maxiter, lambda_, magni, Plot_File, dw);
			}
		}
		else if (model[0]=='T') {
			if (pp==0) {

			}
			else if (pp==1) {
				
			}
		}






		// int lens[6] = {20, 20, 20, 20};
		// TEST_alsCP(6, lens, 8, dw);
		//TEST_sparse_laplacian_alsCP(6, 12, 4, 0, dw); 
		//TEST_sparse_laplacian_alsCP_DT(6, 12, 4, 0, dw); 
		//TEST_sparse_laplacian_alsCP_mod(6, 12, 4, 0, dw); 
		//TEST_dense_uniform_alsCP(100, 5, dw);
 		// ofstream Plot_File("bbb.csv"); 
 		// TEST_3d_poisson_CP(8, 5, 2, 0, 1e-10, 1e-3, 0.00, 1, Plot_File, dw);      
		// TEST_3d_poisson_CP(12, 4, 2, 0, 1e-10, 1e-3, 0.00, 1.0, Plot_File, dw);
		// TEST_poisson_CP(6, 10, 8, 0, 1e-3, 0.00, 0.8, Plot_File, dw);
		// TEST_randmat_CP(6, 14, 5, false, 1e-10, 1e-3, 0.00, 1., Plot_File, dw);
		// TEST_collinearity_CP(6, 14,	5, false, 1e-10, 1e-3, 0.00, 1., 0.5, 0.9, 0.05, Plot_File, dw);


		//TEST_identity_tensor(6, 4, dw);
		//TEST_SVD_solve(6, dw);
		//TEST_laplacian_tensor(4, 8, 1, dw);  // sparse	
		//TEST_gauss_seidel(4, 4, dw);
  // 		ofstream Plot_File("aaa.csv");      
		// TEST_construct_Tucker(6, 10, 2, 0, 1e-10, Plot_File, dw);
  // 		ofstream Plot_File("aaa.csv");      
		// TEST_construct_Tucker_pp(6, 10, 3, 0, 1e-10, 5e-1, Plot_File, dw);

		// int T_lens[] = {13 ,13, 13, 13, 13, 13};
		// int ranks[] = {4, 4, 4, 4, 4, 4};
		//TEST_hosvd(3, T_lens, ranks, dw);
		//TEST_alsTucker(6, T_lens, ranks, dw);	
		//TEST_alsTucker_DT(6, T_lens, ranks, dw);	
		// TEST_alsTucker_mod(6, T_lens, ranks, dw);	
		// TEST_3d_poisson_Tucker(6, 20, 10, 0, dw);
  //   	ofstream Plot_File("poisson_DT_4_36_2_ps.csv");         
		// TEST_3d_poisson_Tucker(8, 6, 2, 0, 1e-10, Plot_File, dw);

  //   	ofstream Plot_File("tucker_dt_4_40_10_ps.csv");      
		// TEST_sparse_laplacian_alsTucker(4, 40, 10, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_4_40_10_ps.csv");       
		// TEST_sparse_laplacian_alsTucker_PP(4, 40, 10, 0, 1e-10, 1e-1, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_14_5_ps.csv");       
		// TEST_sparse_laplacian_alsTucker(6, 14, 5, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_14_5_ps.csv");         
		// TEST_sparse_laplacian_alsTucker_PP(6, 14, 5, 0, 1e-10, 8e-1, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_8_ps.csv");       
		// TEST_sparse_laplacian_alsTucker(6, 16, 8, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_8_ps.csv");         
		// TEST_sparse_laplacian_alsTucker_PP(6, 16, 8, 0, 1e-10, 2e-1, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_5_ps.csv");       
		// TEST_sparse_laplacian_alsTucker(6, 16, 5, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_5_ps.csv");         
		// TEST_sparse_laplacian_alsTucker_PP(6, 16, 5, 0, 1e-10, 1e-2, Plot_File, dw);

  //   	ofstream Plot_File("tucker_dt_4_40_15_rand_ps.csv");      
		// TEST_random_laplacian_alsTucker(4, 40, 15, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_4_40_15_rand_ps.csv");       
		// TEST_random_laplacian_alsTucker_PP(4, 40, 15, 0, 1e-10, 5e-1, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_14_8_rand_ps.csv");       
		// TEST_random_laplacian_alsTucker(6, 14, 8, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_14_8_rand_ps.csv");         
		// TEST_random_laplacian_alsTucker_PP(6, 14, 8, 0, 1e-10, 1e-2, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_8_rand_ps.csv");       
		// TEST_random_laplacian_alsTucker(6, 16, 8, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_8_rand_ps.csv");         
		// TEST_random_laplacian_alsTucker_PP(6, 16, 8, 0, 1e-10, 1e-2, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_5_rand_ps.csv");       
		// TEST_random_laplacian_alsTucker(6, 16, 5, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_5_rand_ps.csv");         
		// TEST_random_laplacian_alsTucker_PP(6, 16, 5, 0, 1e-10, 1e-2, Plot_File, dw);

  //   	ofstream Plot_File("tucker_dt_4_40_10_random.csv");      
		// TEST_random_alsTucker(4, 40, 10, 0, 1e-10, Plot_File, dw);
  //   	ofstream Plot_File("tucker_pp_4_40_10_random.csv");       
		// TEST_random_alsTucker_PP(4, 40, 10, 0, 1e-10, 5e-1, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_14_5_random.csv");       
		// TEST_random_alsTucker(6, 14, 5, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_14_5_random.csv");         
		// TEST_random_alsTucker_PP(6, 14, 5, 0, 1e-10, 1e-2, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_5_random.csv");       
		// TEST_random_alsTucker(6, 16, 5, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_5_random.csv");         
		// TEST_random_alsTucker_PP(6, 16, 5, 0, 1e-10, 1e-2, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_16_8_random.csv");       
		// TEST_random_alsTucker(6, 16, 8, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_16_8_random.csv");         
		// TEST_random_alsTucker_PP(6, 16, 8, 0, 1e-10, 1e-2, Plot_File, dw);
  //   	ofstream Plot_File("tucker_dt_6_13_4_random.csv");       
		// TEST_random_alsTucker(6, 13, 4, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_6_13_4_random.csv");         
		// TEST_random_alsTucker_PP(6, 13, 4, 0, 1e-10, 1e-2, Plot_File, dw);


  //   	ofstream Plot_File("tucker_dt_40_10_uniform.csv");      
		// TEST_dense_uniform_alsTucker(14, 2, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_40_10_uniform.csv");      
		// TEST_dense_uniform_alsTucker_PP(14, 2, 0, 1e-10, 1e-2, Plot_File, dw); 

		// TEST_sparse_laplacian_alsTucker_mod(6, 16, 5, 0, dw); 
		// TEST_sparse_laplacian_alsTucker_mod(6, 20, 7, 0, dw); 
		// 6 16 5 
		// 4 40 10
		// 6, 14, 5, 0, 1e-10, 1e-2


		// int lens_GS[3] = {4, 4, 4};
		// TEST_Gram_Schmidt();
		// TEST_Gen_vector_condition(lens_GS, 3, 2, 1.0);
		//TEST_Gen_tensor_condition(lens_GS, 6, 8, 20, 15, 1.0, dw);
		// TEST_Gen_tensor_condition_pp(lens_GS, 6, 8, 10, 10, 1.0, dw);
		// // 210.309819    227
		// TEST_Gen_tensor_condition_pp(lens_GS, 6, 8, 20, 10, 1.0, dw);
		// 210.309819    227
		// TEST_Gen_tensor_condition_pp(lens_GS, 4, 4, 4, 1, 1.0, dw);
		// TEST_unit_tensor_pp(lens_GS, 3, 4, 1., dw);
	}

	MPI_Finalize();
	return 0;
}

#endif
