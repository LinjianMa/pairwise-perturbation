#include "alscp3.h"
#include "bench.h"
#include "common.h"

#ifndef TEST_SUITE

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
  int partition;
  int bench;
  /*
  r : decomposition of tensor made by random matrices
  */
  int dim;           // number of dimensions
  int s;             // tensor size in each dimension
  int R;             // decomposition rank
  double pp_res_tol; // pp restart tolerance
  double lambda_;    // regularization param
  char *filename;    // output csv filename
  int maxiter = 5e3; // maximum iterations
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
  if (getCmdOption(input_str, input_str + in_num, "-partition")) {
    partition = atoi(getCmdOption(input_str, input_str + in_num, "-partition"));
  } else {
    partition = 0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-bench")) {
    bench = atoi(getCmdOption(input_str, input_str + in_num, "-bench"));
    if (bench != 0)
      bench = 1;
  } else {
    bench = 0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-pp")) {
    pp = atoi(getCmdOption(input_str, input_str + in_num, "-pp"));
    if (pp < 0 || pp > 2)
      pp = 0;
  } else {
    pp = 0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-dim")) {
    dim = atoi(getCmdOption(input_str, input_str + in_num, "-dim"));
    if (dim < 0)
      dim = 3;
  } else {
    dim = 3;
  }
  if (getCmdOption(input_str, input_str + in_num, "-maxiter")) {
    maxiter = atoi(getCmdOption(input_str, input_str + in_num, "-maxiter"));
    if (maxiter < 0)
      maxiter = 2000;
  } else {
    maxiter = 2000;
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
      cout << "  tensor=  " << tensor << "  pp=  " << pp << endl;
      cout << "  dim=  " << dim << "  size=  " << s << "  rank=  " << R << endl;
      cout << "  restarttol=  " << pp_res_tol << endl;
      cout << "  lambda=  " << lambda_ << "  filename=  " << filename << endl;
      cout << "  maxiter=  " << maxiter << "  resprint=  " << resprint << endl;
      cout << "  tensorfile=  " << tensorfile << endl;
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
    } else if (tensor[0] == 's') {
      // s1 : 3 water molecules, 339 x 21 x 21
      if (strlen(tensor) > 1 && tensor[1] == '1') {
        tensorfile = "../bin/scf-3.bin";
        MPI_File_open(MPI_COMM_WORLD, tensorfile,
                      MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        int lens[3] = {21, 21, 339};
        V = Tensor<>(dim, lens, dw);
        if (dw.rank == 0)
          cout << "Read the tensor from file scf-3.bin ...... " << endl;
        V.read_dense_from_file(fh);
        if (dw.rank == 0)
          cout << "Read scf-3.bin dataset finished " << endl;
        // V.print();
      }
      // s2 : 40 water molecules, 4520 x 280 x 280
      else if (strlen(tensor) > 1 && tensor[1] == '2') {
        tensorfile = "../bin/scf-40.bin";
        MPI_File_open(MPI_COMM_WORLD, tensorfile,
                      MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        int lens[3] = {280, 280, 4520};
        V = Tensor<>(dim, lens, dw);
        if (dw.rank == 0)
          cout << "Read the tensor from file scf-40.bin ...... " << endl;
        V.read_dense_from_file(fh);
        if (dw.rank == 0)
          cout << "Read scf-40.bin dataset finished " << endl;
        // V.print();
      }
    }

    double Vnorm = V.norm2();
    if (dw.rank == 0)
      cout << "Vnorm= " << Vnorm << endl;
    ofstream Plot_File(filename);
    Matrix<> *W = new Matrix<>[V.order]; // N matrices V will be decomposed into
    for (int i = 0; i < V.order; i++) {
      W[i] = Matrix<>(V.lens[i], R, dw);
      W[i].fill_random(0, 1);
    }

    // V.write_dense_to_file (fh);

    Timer_epoch tALS("ALS");
    tALS.begin();

    // bench_contraction(s, 5, "ijk", "ia", "ajk", dw);
    // bench_contraction(s, 5, "ijk", "ia", "jka", dw);
    // bench_contraction(s, 5, "ijk", "ja", "aik", dw);
    // bench_contraction(s, 5, "ijk", "ja", "ika", dw);
    // bench_contraction(s, 5, "ijk", "ka", "aij", dw);
    // bench_contraction(s, 5, "ijk", "ka", "ija", dw);

    // bench_contraction(s, 5, "ajk", "aj", "ak", dw);
    // bench_contraction_no_dist(s, 5, "ajk", "aj", "ak", dw);
    // bench_contraction(s, 5, "aik", "ak", "ai", dw);
    // bench_contraction_no_dist(s, 5, "aik", "ak", "ai", dw);
    // bench_contraction(s, 5, "aij", "ai", "aj", dw);
    // bench_contraction_no_dist(s, 5, "aij", "ai", "aj", dw);

    // bench_contraction(s, 5, "jka", "ja", "ka", dw);
    // bench_contraction_no_dist(s, 5, "jka", "ja", "ka", dw);
    // bench_contraction(s, 5, "ika", "ka", "ia", dw);
    // bench_contraction_no_dist(s, 5, "ika", "ka", "ia", dw);
    // bench_contraction(s, 5, "ija", "ia", "ja", dw);
    // bench_contraction_no_dist(s, 5, "ija", "ia", "ja", dw);

    cout.precision(13);
    if (dw.rank == 0) {
      Plot_File << "[dim],[iter],[fitness],[pp_update],[diffV],[dtime]"
                << "\n"; // Headings for file
    }
    if (bench == 1) {
      alscp_pp3_bench(V, W, maxiter, pp_res_tol, lambda_, Plot_File, resprint,
                      partition, dw);
    } else {
      if (pp == 0) {
        alscp_dt3(V, W, maxiter, lambda_, Plot_File, resprint, partition, dw);
      } else if (pp == 1) {
        alscp_pp3(V, W, maxiter, pp_res_tol, lambda_, Plot_File, resprint,
                  partition, dw);
      }
    }

    tALS.end();

    if (dw.rank == 0) {
      printf("experiment took %lf seconds\n", MPI_Wtime() - start_time);
    }

    delete[] W;

    if (tensor[0] == 'o') {
      MPI_File_close(&fh);
    }
  }

  MPI_Finalize();
  return 0;
}

#endif
