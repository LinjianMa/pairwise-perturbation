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
	//int const in_num = argc;
	//char ** input_str = argv;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	{
		World dw(argc, argv);

		srand48(dw.rank*2+13);

		int lens[6] = {20, 20, 20, 20};
		// TEST_alsCP(6, lens, 8, dw);
		//TEST_sparse_laplacian_alsCP(6, 12, 4, 0, dw); 
		//TEST_sparse_laplacian_alsCP_DT(6, 12, 4, 0, dw); 
		//TEST_sparse_laplacian_alsCP_mod(6, 12, 4, 0, dw); 
		//TEST_dense_uniform_alsCP(100, 5, dw);
		//TEST_3d_poisson_CP(6, 3, 3, 0, dw);

		//TEST_identity_tensor(6, 4, dw);
		//TEST_SVD_solve(6, dw);
		//TEST_laplacian_tensor(4, 8, 1, dw);  // sparse	
		//TEST_gauss_seidel(4, 4, dw);
  		ofstream Plot_File("aaa.csv");      
		TEST_construct_Tucker(6, 10, 2, 0, 1e-10, Plot_File, dw);
  // 		ofstream Plot_File("aaa.csv");      
		// TEST_construct_Tucker_pp(6, 10, 3, 0, 1e-10, 5e-1, Plot_File, dw);

		int T_lens[] = {13 ,13, 13, 13, 13, 13};
		int ranks[] = {4, 4, 4, 4, 4, 4};
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


		int lens_GS[3] = {4, 4, 4};
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
