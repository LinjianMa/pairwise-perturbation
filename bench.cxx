
#include "bench.h"
//#define ERR_REPORT

int bench_contraction(int n, int niter, char const *iA, char const *iB,
                      char const *iC, CTF_World &dw) {

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

  for (i = 0; i < order_A; i++) {
    n_A[i] = n;
  }
  for (i = 0; i < order_B; i++) {
    n_B[i] = n;
  }
  for (i = 0; i < order_C; i++) {
    n_C[i] = n;
  }

  Tensor<> A(order_A, n_A, dw);
  Tensor<> B(order_B, n_B, dw);
  Tensor<> C(order_C, n_C, dw);

  double st_time = MPI_Wtime();

  for (i = 0; i < niter; i++) {
    C[iC] += A[iA] * B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf "
           "sec/iter\n",
           niter, iC, iA, iB, (end_time - st_time) / niter);

  return 1;
}

int bench_contraction_no_dist(int n, int niter, char const *iA, char const *iB,
                              char const *iC, CTF_World &dw) {

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

  for (i = 0; i < order_A; i++) {
    n_A[i] = n;
  }
  for (i = 0; i < order_B; i++) {
    n_B[i] = n;
  }
  for (i = 0; i < order_C; i++) {
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

  for (i = 0; i < niter; i++) {
    C[iC] += A[iA] * B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf "
           "sec/iter\n",
           niter, iC, iA, iB, (end_time - st_time) / niter);

  return 1;
}
