/** \addtogroup examples 
  * @{ 
  * \defgroup TESTS_multigrid TESTS_multigrid
  * @{ 
  * \brief NTF/TF multigrid tests
  */
#include "als_Tucker.h"
#include "als_CP.h"
#include "common.h"
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
	double timelimit = 5e3;  // time limits
	int maxiter = 5e3;		// maximum iterations
	int resprint = 1;

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
    	if (dim < 0) dim = 8;
	} else {
		dim = 8;
	}
	if (getCmdOption(input_str, input_str+in_num, "-maxiter")) {
		maxiter = atoi(getCmdOption(input_str, input_str+in_num, "-maxiter"));
    	if (maxiter < 0) maxiter = 5e3;
	} else {
		maxiter = 5e3;
	}
	if (getCmdOption(input_str, input_str+in_num, "-timelimit")) {
		timelimit = atof(getCmdOption(input_str, input_str+in_num, "-timelimit"));
    	if (timelimit < 0) timelimit = 5e3;
	} else {
		timelimit = 5e3;
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
	if (getCmdOption(input_str, input_str+in_num, "-resprint")) {
		resprint = atoi(getCmdOption(input_str, input_str+in_num, "-resprint"));
    	if (resprint < 0) resprint = 10;
	} else {
		resprint = 10;
	}
	if (getCmdOption(input_str, input_str+in_num, "-tol")) {
		tol = atof(getCmdOption(input_str, input_str+in_num, "-tol"));
    	if (tol < 0 || tol > 1) tol = 1e-10;
	} else {
		tol = 1e-10;
	}	
	if (getCmdOption(input_str, input_str+in_num, "-pp_res_tol")) {
		pp_res_tol = atof(getCmdOption(input_str, input_str+in_num, "-pp_res_tol"));
    	if (pp_res_tol < 0 || pp_res_tol > 1) pp_res_tol = 1e-2;
	} else {
		pp_res_tol = 1e-2;
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
		double start_time = MPI_Wtime();
		World dw(argc, argv);
		srand48(dw.rank*1);

		if (dw.rank==0) {
			cout << "  model=  " << model << "  tensor=  " << tensor << "  pp=  " << pp << endl;
			cout << "  dim=  " << dim << "  size=  " << s << "  rank=  " << R << endl;
			cout << "  issparse=  " << issparse << "  tolerance=  " << tol << "  restarttol=  " << pp_res_tol << endl;
			cout << "  lambda=  " << lambda_ << "  magnitude=  " << magni << "  filename=  " << filename << endl;
			cout << "  col_min=  " << col_min << "  col_max=  " << col_max  << "  rationoise  " << ratio_noise << endl;
			cout << "  timelimit=  " << timelimit << "  maxiter=  " << maxiter << "  resprint=  " << resprint  << endl;
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
					W[i].fill_random(-0,1); 
				}
				build_V(V, W, dim, dw);
				delete[] W;
			}
		}
		else if (tensor[0]=='o') {
			//o : other tensor
			// TODO
		}

		double Vnorm = V.norm2();
 		ofstream Plot_File(filename); 
 		Matrix<>* W = new Matrix<>[V.order];
		Matrix<>* W_DT = new Matrix<>[V.order];				// N matrices V will be decomposed into
		Matrix<>* W_PP = new Matrix<>[V.order];
		Matrix<>* grad_W = new Matrix<>[V.order];			// gradients in N dimensions 
		for (int i=0; i<V.order; i++) {
			W[i] = Matrix<>(V.lens[i],R,dw);
			W_DT[i] = Matrix<>(V.lens[i],R,dw);
			W_PP[i] = Matrix<>(V.lens[i],R,dw);
			grad_W[i] = Matrix<>(V.lens[i],R,dw);
			W[i].fill_random(0,1);
			W_DT[i]["ij"] = W[i]["ij"]; 
			W_PP[i]["ij"] = W[i]["ij"];
			grad_W[i].fill_random(0,1);  
		}
		//construct F matrices (correction terms, F[]=0 initially)
		Matrix<>* F = new Matrix<>[V.order];
		for (int i=0; i<V.order; i++) {
			F[i] = Matrix<>(V.lens[i],R,dw);
			F[i]["ij"] = 0.;
		}
    Timer_epoch tALS("ALS");
    tALS.begin();
		if (model[0]=='C') {
    		if (dw.rank==0) Plot_File << "[timetype],[dtime]" << "\n";          //Headings for file
			for (int i=0; i<maxiter; i++) {
				alsCP_DT(V, W_DT, grad_W, F, tol*Vnorm, timelimit, 1, lambda_, Plot_File, resprint, true, dw);
				for (int j=0; j<V.order; j++) {
					W_DT[j]["ij"] = W[j]["ij"];
				}
			}
			if (dw.rank==0) Plot_File << endl;
			for (int i=0; i<maxiter; i++) {
				alsCP_PP(V, W_PP, grad_W, F, tol*Vnorm, pp_res_tol, timelimit, 1, lambda_, magni, Plot_File, resprint, true, dw);
				for (int j=0; j<V.order; j++) {
					W_PP[j]["ij"] = W[j]["ij"];
				}
			}
			if (dw.rank==0) Plot_File << endl;
		}
		else if (model[0]=='T') {
    		if (dw.rank==0) Plot_File << "[timetype],[dtime]" << "\n";          //Headings for file
			int ranks[V.order];
			for (int i=0; i<V.order; i++) {
				ranks[i] = R;
			}
			// using hosvd to initialize W and hosvd_core
			Tensor<> hosvd_core = Tensor<>(dim, issparse, ranks, dw); 
			// hosvd(V, hosvd_core, W, ranks, dw);
			for (int i=0; i<maxiter; i++) {
				for (int j=0; j<V.order; j++){
					W_DT[j]["ij"] = W[j]["ij"];
				}
				// Tensor<> hosvdcore_DT(hosvd_core);
				alsTucker_DT(V, hosvd_core, W_DT, tol*Vnorm, timelimit, 1, Plot_File, resprint, true, dw);
			}
			if (dw.rank==0) Plot_File << endl;
			for (int i=0; i<maxiter; i++) {
				for (int j=0; j<V.order; j++){
					W_PP[j]["ij"] = W[j]["ij"];
				}
				// Tensor<> hosvdcore_PP(hosvd_core);
				alsTucker_PP(V, hosvd_core, W_PP, tol*Vnorm, pp_res_tol, timelimit, 1, Plot_File, resprint, true, dw);	
			}	
			if (dw.rank==0) Plot_File << endl;
		}
    tALS.end();

		if(dw.rank==0) {
			printf ("experiment took %lf seconds\n",MPI_Wtime()-start_time);
		}

		delete[] F;
 		delete[] W;
		delete[] W_DT;				
		delete[] W_PP;
		delete[] grad_W;

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

  //   	ofstream Plot_File("tucker_dt_40_10_uniform.csv");      
		// TEST_dense_uniform_alsTucker(14, 2, 0, 1e-10, Plot_File, dw); 
  //   	ofstream Plot_File("tucker_pp_40_10_uniform.csv");      
		// TEST_dense_uniform_alsTucker_PP(14, 2, 0, 1e-10, 1e-2, Plot_File, dw); 

		// TEST_sparse_laplacian_alsTucker_mod(6, 16, 5, 0, dw); 
		// TEST_sparse_laplacian_alsTucker_mod(6, 20, 7, 0, dw); 
		// 6 16 5 
		// 4 40 10
		// 6, 14, 5, 0, 1e-10, 1e-2
	}

	MPI_Finalize();
	return 0;
}

#endif
