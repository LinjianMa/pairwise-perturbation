#include "alscp3.h"
#include "common.h"

bool alscp_dt3(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, double tol,
               double timelimit, int maxiter, double lambda,
               ofstream &Plot_File, int resprint, bool bench, World &dw) {
  cout.precision(13);
  if (dw.rank == 0)
    Plot_File << "[dim],[iter],[fitness],[tol],[pp_update],[diffV],[dtime]"
              << "\n"; // Headings for file

  double Vnorm = V.norm2();

  int rank = W[0].ncol;
  //  initialize matrix S
  Matrix<> S = Matrix<>(rank, rank);
  Matrix<> regul = Matrix<>(rank, rank);
  regul["ii"] = 1. * lambda;

  Matrix<> AA = Matrix<>(rank, rank);
  Matrix<> BB = Matrix<>(rank, rank);
  Matrix<> CC = Matrix<>(rank, rank);

  int dim1 = V.lens[0];
  int dim2 = V.lens[1];
  int dim3 = V.lens[2];

  int order_Tc[3] = {dim1, dim2, rank};
  int order_Ta[3] = {dim2, dim3, rank};
  // Tensor<> T_C = Tensor<>(3, order_Tc, dw);
  // Tensor<> T_A = Tensor<>(3, order_Ta, dw);
  int np = dw.np;
  int syms[3] = {NS, NS, NS};
  CTF::Partition p(1, &np);
  Tensor<> T_C = Tensor<>(3, order_Tc, syms, dw, "ija", p["a"]);
  Tensor<> T_A = Tensor<>(3, order_Ta, syms, dw, "ija", p["a"]);

  Matrix<> T_BC = Matrix<>(V.lens[0], rank);
  Matrix<> T_AC = Matrix<>(V.lens[1], rank);
  Matrix<> T_AB = Matrix<>(V.lens[2], rank);

  double st_time = MPI_Wtime();
  int iter;
  double Fnorm = 0.;
  double diffnorm_V = 1000;

  for (iter = 0; iter <= maxiter; iter++) {
    // print the gradient norm
    if (iter % resprint == 0 || iter == maxiter) {
      double st_time1 = MPI_Wtime();
      // diffnorm
      Tensor<> V_build;
      build_V(V_build, W, V.order, dw);
      Tensor<> diff_V = V;
      diff_V["ijk"] = V["ijk"] - V_build["ijk"];
      diffnorm_V = diff_V.norm2();
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      double fitness = 1. - diffnorm_V / Vnorm;
      if (dw.rank == 0) {
        cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter
             << "  [fitness]  " << fitness << "  [tol]  " << tol
             << "  [pp_update]  " << 0 << "  [diffV]  " << diffnorm_V
             << "  [dtime]  " << dtime << "\n";
        Plot_File << V.lens[0] << "," << iter << "," << fitness << "," << tol
                  << "," << 0 << "," << diffnorm_V << "," << dtime << "\n";
        if (iter % 100 == 0 && iter != 0) { // flush
          Plot_File << endl;
        }
      }
    }
    // iteration on W[i]
    for (int i = 0; i < V.order; i++) {
      // W[0]
      double time_contract1 = MPI_Wtime();
      T_C["ij*"] = V["ijk"] * W[2]["k*"];
      double time_contract2 = MPI_Wtime();
      T_BC["i*"] = T_C["ij*"] * W[1]["j*"];
      double time_contract3 = MPI_Wtime();

      if (dw.rank == 0) {
        printf("\nfirst level contraction %lf second level contraction %lf \n",
               time_contract2 - time_contract1,
               time_contract3 - time_contract2);
      }

      BB["ij"] = W[1]["ki"] * W[1]["kj"];
      CC["ij"] = W[2]["ki"] * W[2]["kj"];

      S["ij"] = BB["ij"] * CC["ij"];
      S["ij"] += regul["ij"];
      SVD_solve(T_BC, W[0], S);

      // W[1]
      T_AC["j*"] = T_C["ij*"] * W[0]["i*"];
      AA["ij"] = W[0]["ki"] * W[0]["kj"];

      S["ij"] = AA["ij"] * CC["ij"];
      S["ij"] += regul["ij"];

      SVD_solve(T_AC, W[1], S);

      // W[2]
      time_contract1 = MPI_Wtime();
      T_A["jk*"] = V["ijk"] * W[0]["i*"];
      time_contract2 = MPI_Wtime();
      T_AB["k*"] = T_A["jk*"] * W[1]["j*"];
      time_contract3 = MPI_Wtime();

      if (dw.rank == 0) {
        printf("\nfirst level contraction %lf second level contraction %lf \n",
               time_contract2 - time_contract1,
               time_contract3 - time_contract2);
      }

      BB["ij"] = W[1]["ki"] * W[1]["kj"];

      S["ij"] = AA["ij"] * BB["ij"];
      S["ij"] += regul["ij"];
      SVD_solve(T_AB, W[2], S);
    }
    // print .
    if (iter % 10 == 0 && dw.rank == 0)
      printf(".");
  }
  if (dw.rank == 0) {
    printf("tf took %lf seconds\n", MPI_Wtime() - st_time);
  }
  if (bench == false) {
    Plot_File.close();
  }
  if (iter == maxiter + 1)
    return false;
  else
    return true;
}
