#include "common.h"

#ifndef TEST_SUITE

int bench_contraction(int          n,
                      int          niter,
                      char const * iA,
                      char const * iB,
                      char const * iC,
                      CTF_World   &dw){

  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int order_A, order_B, order_C;
  order_A = strlen(iA);
  order_B = strlen(iB);
  order_C = strlen(iC);

  // int NS_A[order_A];
  // int NS_B[order_B];
  // int NS_C[order_C];
  int n_A[order_A];
  int n_B[order_B];
  int n_C[order_C];

  for (i=0; i<order_A; i++){
    n_A[i] = n;
    // NS_A[i] = NS;
  }
  for (i=0; i<order_B; i++){
    n_B[i] = n;
    // NS_B[i] = NS;
  }
  for (i=0; i<order_C; i++){
    n_C[i] = n;
    // NS_C[i] = NS;
  }


  //* Creates distributed tensors initialized with zeros
  // CTF_Tensor A(order_A, n_A, NS_A, dw, "A", 1);
  // CTF_Tensor B(order_B, n_B, NS_B, dw, "B", 1);
  // CTF_Tensor C(order_C, n_C, NS_C, dw, "C", 1);

  Tensor<> A(order_A, n_A, dw);
  Tensor<> B(order_B, n_B, dw);
  Tensor<> C(order_C, n_C, dw);

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    C[iC] += A[iA]*B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf sec/iter\n", 
           niter, iC, iA, iB, (end_time-st_time)/niter);

  return 1;
} 

int bench_contraction_no_dist(int          n,
                      int          niter,
                      char const * iA,
                      char const * iB,
                      char const * iC,
                      CTF_World   &dw){

  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int order_A, order_B, order_C;
  order_A = strlen(iA);
  order_B = strlen(iB);
  order_C = strlen(iC);

  int n_A[order_A];
  int n_B[order_B];
  int n_C[order_C];

  for (i=0; i<order_A; i++){
    n_A[i] = n;
  }
  for (i=0; i<order_B; i++){
    n_B[i] = n;
  }
  for (i=0; i<order_C; i++){
    n_C[i] = n;
  }

  Tensor<> A;
  Tensor<> B(order_B, n_B, dw);
  Tensor<> C(order_C, n_C, dw);

  int np = dw.np;
  int syms[3] = {NS, NS, NS};
  CTF::Partition p(1, &np);
  A = Tensor<>(order_A, n_A, syms, dw, "ija", p["a"]);

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    C[iC] += A[iA]*B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf sec/iter\n", 
           niter, iC, iA, iB, (end_time-st_time)/niter);

  return 1;
} 

char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int rank, np; //, n, pass;
  int const in_num = argc;
  char **input_str = argv;

  char *tensor; // which tensor    p / p2 / c / r / r2 / o /
  int pp;       // 0 Dimention tree 1 pairwise perturbation 2 pp with <1
                // update_percentage_pp
  double update_percentage_pp; // pp update ratio. For each sweep only update
                               // update_percentage_pp*N matrices.
  /*
  r : decomposition of tensor made by random matrices
  */
  int dim;                // number of dimensions
  int s;                  // tensor size in each dimension
  int R;                  // decomposition rank
  double tol;             // global convergance tolerance
  double pp_res_tol;      // pp restart tolerance
  double lambda_;         // regularization param
  double magni;           // pp update magnitude
  char *filename;         // output csv filename
  double timelimit = 5e3; // time limits
  int maxiter = 5e3;      // maximum iterations
  int resprint = 1;
  char *tensorfile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_File fh;

  if (getCmdOption(input_str, input_str + in_num, "-tensor")) {
    tensor = getCmdOption(input_str, input_str + in_num, "-tensor");
  } else {
    tensor = "r";
  }
  if (getCmdOption(input_str, input_str + in_num, "-pp")) {
    pp = atoi(getCmdOption(input_str, input_str + in_num, "-pp"));
    if (pp < 0 || pp > 2)
      pp = 0;
  } else {
    pp = 0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-update_percentage_pp")) {
    update_percentage_pp = atof(
        getCmdOption(input_str, input_str + in_num, "-update_percentage_pp"));
    if (update_percentage_pp < 0 || update_percentage_pp > 1)
      update_percentage_pp = 1.0;
  } else {
    update_percentage_pp = 1.0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-dim")) {
    dim = atoi(getCmdOption(input_str, input_str + in_num, "-dim"));
    if (dim < 0)
      dim = 8;
  } else {
    dim = 8;
  }
  if (getCmdOption(input_str, input_str + in_num, "-maxiter")) {
    maxiter = atoi(getCmdOption(input_str, input_str + in_num, "-maxiter"));
    if (maxiter < 0)
      maxiter = 5e3;
  } else {
    maxiter = 5e3;
  }
  if (getCmdOption(input_str, input_str + in_num, "-timelimit")) {
    timelimit = atof(getCmdOption(input_str, input_str + in_num, "-timelimit"));
    if (timelimit < 0)
      timelimit = 5e3;
  } else {
    timelimit = 5e3;
  }
  if (getCmdOption(input_str, input_str + in_num, "-size")) {
    s = atoi(getCmdOption(input_str, input_str + in_num, "-size"));
    if (s < 0)
      s = 10;
  } else {
    s = 10;
  }
  if (getCmdOption(input_str, input_str + in_num, "-rank")) {
    R = atoi(getCmdOption(input_str, input_str + in_num, "-rank"));
    if (R < 0 || R > s)
      R = s / 2;
  } else {
    R = s / 2;
  }
  if (getCmdOption(input_str, input_str + in_num, "-resprint")) {
    resprint = atoi(getCmdOption(input_str, input_str + in_num, "-resprint"));
    if (resprint < 0)
      resprint = 10;
  } else {
    resprint = 10;
  }
  if (getCmdOption(input_str, input_str + in_num, "-tol")) {
    tol = atof(getCmdOption(input_str, input_str + in_num, "-tol"));
    if (tol < 0 || tol > 1)
      tol = 1e-10;
  } else {
    tol = 1e-10;
  }
  if (getCmdOption(input_str, input_str + in_num, "-pp_res_tol")) {
    pp_res_tol =
        atof(getCmdOption(input_str, input_str + in_num, "-pp_res_tol"));
    if (pp_res_tol < 0 || pp_res_tol > 1)
      pp_res_tol = 1e-2;
  } else {
    pp_res_tol = 1e-2;
  }
  if (getCmdOption(input_str, input_str + in_num, "-lambda")) {
    lambda_ = atof(getCmdOption(input_str, input_str + in_num, "-lambda"));
    if (lambda_ < 0)
      lambda_ = 0.;
  } else {
    lambda_ = 0.;
  }
  if (getCmdOption(input_str, input_str + in_num, "-magni")) {
    magni = atof(getCmdOption(input_str, input_str + in_num, "-magni"));
    if (magni < 0)
      magni = 1.;
  } else {
    magni = 1.;
  }
  if (getCmdOption(input_str, input_str + in_num, "-filename")) {
    filename = getCmdOption(input_str, input_str + in_num, "-filename");
  } else {
    filename = "out.csv";
  }
  if (getCmdOption(input_str, input_str + in_num, "-tensorfile")) {
    tensorfile = getCmdOption(input_str, input_str + in_num, "-tensorfile");
  } else {
    tensorfile = "test";
  }

  {
    double start_time = MPI_Wtime();
    World dw(argc, argv);
    srand48(dw.rank * 1);

    if (dw.rank == 0) {
      cout << "  tensor=  " << tensor
           << "  pp=  " << pp << endl;
      cout << "  dim=  " << dim << "  size=  " << s << "  rank=  " << R << endl;
      cout << "  tolerance=  " << tol
           << "  restarttol=  " << pp_res_tol << endl;
      cout << "  lambda=  " << lambda_ << "  magnitude=  " << magni
           << "  filename=  " << filename << endl;
      cout << "  timelimit=  " << timelimit << "  maxiter=  " << maxiter
           << "  resprint=  " << resprint << endl;
      cout << "  tensorfile=  " << tensorfile
           << "  update_percentage_pp=  " << update_percentage_pp << endl;
    }

    // initialization of tensor
    Tensor<> V;

    if (tensor[0] == 'r') {
        // r : tensor made by random matrices
        int lens[dim];
        for (int i = 0; i < dim; i++)
          lens[i] = s;
        Matrix<> *W = new Matrix<>[dim]; // N matrices V will be decomposed into
        for (int i = 0; i < dim; i++) {
          W[i] = Matrix<>(s, R, dw);
          W[i].fill_random(0, 1);
        }
        build_V(V, W, dim, dw);
        delete[] W;
      }

    double Vnorm = V.norm2();
    if (dw.rank == 0)
      cout << "Vnorm= " << Vnorm << endl;
    ofstream Plot_File(filename);
    Matrix<> *W = new Matrix<>[V.order]; // N matrices V will be decomposed into
    Matrix<> *grad_W = new Matrix<>[V.order]; // gradients in N dimensions
    for (int i = 0; i < V.order; i++) {
      W[i] = Matrix<>(V.lens[i], R, dw);
      grad_W[i] = Matrix<>(V.lens[i], R, dw);
      W[i].fill_random(0, 1);
      grad_W[i].fill_random(0, 1);
    }
    // construct F matrices (correction terms, F[]=0 initially)
    Matrix<> *F = new Matrix<>[V.order];
    for (int i = 0; i < V.order; i++) {
      F[i] = Matrix<>(V.lens[i], R, dw);
      F[i]["ij"] = 0.;
    }

    // V.write_dense_to_file (fh);

    Timer_epoch tALS("ALS");
    tALS.begin();

    bench_contraction(s, 5, "ijk", "ia", "ajk", dw);
    bench_contraction(s, 5, "ijk", "ia", "jka", dw);
    bench_contraction(s, 5, "ijk", "ja", "aik", dw);
    bench_contraction(s, 5, "ijk", "ja", "ika", dw);
    bench_contraction(s, 5, "ijk", "ka", "aij", dw);
    bench_contraction(s, 5, "ijk", "ka", "ija", dw);

    // bench_contraction(s, 5, "ajk", "aj", "ak", dw);
    // bench_contraction_no_dist(s, 5, "ajk", "aj", "ak", dw);
    // bench_contraction(s, 5, "aik", "ak", "ai", dw);
    // bench_contraction_no_dist(s, 5, "aik", "ak", "ai", dw);
    // bench_contraction(s, 5, "aij", "ai", "aj", dw);
    // bench_contraction_no_dist(s, 5, "aij", "ai", "aj", dw);

    bench_contraction(s, 5, "jka", "ja", "ka", dw);
    bench_contraction_no_dist(s, 5, "jka", "ja", "ka", dw);
    bench_contraction(s, 5, "ika", "ka", "ia", dw);
    bench_contraction_no_dist(s, 5, "ika", "ka", "ia", dw);
    bench_contraction(s, 5, "ija", "ia", "ja", dw);
    bench_contraction_no_dist(s, 5, "ija", "ia", "ja", dw);

    // int lens_V[3];
    // lens_V[0] = V.lens[0];
    // lens_
    // lens_V[i + 1] = W[0].ncol;
    // Tensor<> V_temp = Tensor<>(i + 2, lens_V, dw);

    // // benchmark MTTKRP
    // double mttkrp_start_time = MPI_Wtime();
    // if (dw.rank == 0) {
    //   printf("experiment took %lf seconds\n", MPI_Wtime() - mttkrp_start_time);
    // }


      // if (pp == 0) {
      //   alsCP_DT(V, W, grad_W, F, tol * Vnorm, timelimit, maxiter, lambda_,
      //            Plot_File, resprint, false, dw);
      // } else if (pp == 1) {
      //   alsCP_PP(V, W, grad_W, F, tol * Vnorm, pp_res_tol, timelimit, maxiter,
      //            lambda_, magni, Plot_File, resprint, false, dw);
      // } else if (pp == 2) {
      //   alsCP_PP_partupdate(V, W, grad_W, F, tol * Vnorm, pp_res_tol, timelimit,
      //                       maxiter, lambda_, magni, update_percentage_pp,
      //                       Plot_File, resprint, false, dw);
      // }

    tALS.end();

    if (dw.rank == 0) {
      printf("experiment took %lf seconds\n", MPI_Wtime() - start_time);
    }

    delete[] W;
    delete[] grad_W;
    delete[] F;

    if (tensor[0] == 'o') {
      MPI_File_close(&fh);
    }
  }

  MPI_Finalize();
  return 0;
}

#endif
