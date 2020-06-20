#include "alscp3.h"
#include "common.h"

bool alscp_dt3(Tensor<> &V, Matrix<> *W, int maxiter, double lambda,
               ofstream &Plot_File, int resprint, int partition, World &dw, double Vnorm) {
  // Timer_epoch tALS("alscp_dt3");
  // tALS.begin();

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

  Tensor<> T_C = Tensor<>(3, order_Tc, dw);
  Tensor<> T_A = Tensor<>(3, order_Ta, dw);
  if (partition != 0) {
    if (dw.rank == 0) {
      cout << "partition is 1" << endl;
    }
    int np = dw.np;
    int syms[3] = {NS, NS, NS};
    CTF::Partition p(1, &np);
    T_C = Tensor<>(3, order_Tc, syms, dw, "ija", p["a"]);
    T_A = Tensor<>(3, order_Ta, syms, dw, "ija", p["a"]);
  }

  Matrix<> T_BC = Matrix<>(rank, V.lens[0]);
  Matrix<> T_AC = Matrix<>(rank, V.lens[1]);
  Matrix<> T_AB = Matrix<>(rank, V.lens[2]);

  double st_time = MPI_Wtime();
  int iter;
  double diffnorm_V = 1000;

  for (iter = 0; iter <= maxiter; iter++) {
    // W[0]
    T_C["ij*"] = V["ijk"] * W[2]["k*"];
    T_BC["*i"] = T_C["ij*"] * W[1]["j*"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];
    CC["ij"] = W[2]["ki"] * W[2]["kj"];

    S["ij"] = BB["ij"] * CC["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W0_trans = Matrix<>(rank, V.lens[0]);
    T_BC.solve_spd(S, W0_trans);
    W[0]["ij"] = W0_trans["ji"];

    // W[1]
    T_AC["*j"] = T_C["ij*"] * W[0]["i*"];
    AA["ij"] = W[0]["ki"] * W[0]["kj"];

    S["ij"] = AA["ij"] * CC["ij"];
    S["ij"] += regul["ij"];

    Matrix<> W1_trans = Matrix<>(rank, V.lens[1]);
    T_AC.solve_spd(S, W1_trans);
    W[1]["ij"] = W1_trans["ji"];

    // W[2]
    T_A["jk*"] = V["ijk"] * W[0]["i*"];
    T_AB["*k"] = T_A["jk*"] * W[1]["j*"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];

    S["ij"] = AA["ij"] * BB["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W2_trans = Matrix<>(rank, V.lens[2]);
    T_AB.solve_spd(S, W2_trans);
    W[2]["ij"] = W2_trans["ji"];

    // print .
    if (iter % 10 == 0 && dw.rank == 0)
      printf(".");

    // residual calculation
    if (iter % resprint == 0 || iter == maxiter) {
      double st_time1 = MPI_Wtime();
      // diffnorm
      Matrix<> AA = Matrix<>(rank, rank);
      Matrix<> BB = Matrix<>(rank, rank);
      Matrix<> CC = Matrix<>(rank, rank);
      AA["ab"] = W[0]["ka"] * W[0]["kb"];
      BB["ab"] = W[1]["ka"] * W[1]["kb"];
      CC["ab"] = W[2]["ka"] * W[2]["kb"];
      double inner_W = AA["ab"] * BB["ab"] * CC["ab"];
      T_AB["*k"] = T_A["jk*"] * W[1]["j*"];
      double T_ABC = T_AB["ab"] * W[2]["ba"];
      diffnorm_V = sqrt(Vnorm * Vnorm + inner_W - 2. * T_ABC);
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      double fitness = 1. - diffnorm_V / Vnorm;
      if (dw.rank == 0) {
        cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter
             << "  [fitness]  " << fitness << "  [pp_update]  " << 0
             << "  [diffV]  " << diffnorm_V << "  [dtime]  " << dtime << "\n";
        Plot_File << V.lens[0] << "," << iter << "," << fitness << "," << 0
                  << "," << diffnorm_V << "," << dtime << "\n";
        if (iter % 100 == 0 && iter != 0) { // flush
          Plot_File << endl;
        }
      }
    }
  }
  if (dw.rank == 0) {
    printf("tf took %lf seconds\n", MPI_Wtime() - st_time);
  }
  Plot_File.close();
  // tALS.end();
  if (iter == maxiter + 1)
    return false;
  else
    return true;
}

void alscp_dt3_sub(Tensor<> &V, Matrix<> *W, Matrix<> *dW, Tensor<> &T_A, double tol_init,
                   int maxiter, double &st_time, double lambda,
                   ofstream &Plot_File, int &iter, int resprint, World &dw, double Vnorm) {
  // Timer_epoch tALS("alscp_dt3");
  // tALS.begin();

  Matrix<> *W_prev = new Matrix<>[V.order];
  for (int i = 0; i < V.order; i++) {
    W_prev[i] = Matrix<>(W[i].nrow, W[i].ncol);
    W_prev[i]["ij"] = W[i]["ij"];
  }

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
  Tensor<> T_C = Tensor<>(3, order_Tc, dw);
  T_A = Tensor<>(3, order_Ta, dw);
  // int np = dw.np;
  // int syms[3] = {NS, NS, NS};
  // CTF::Partition p(1, &np);
  // Tensor<> T_C = Tensor<>(3, order_Tc, syms, dw, "ija", p["a"]);
  // Tensor<> T_A = Tensor<>(3, order_Ta, syms, dw, "ija", p["a"]);

  Matrix<> T_BC = Matrix<>(rank, V.lens[0]);
  Matrix<> T_AC = Matrix<>(rank, V.lens[1]);
  Matrix<> T_AB = Matrix<>(rank, V.lens[2]);

  double diffnorm_V = 1000;

  for (; iter <= maxiter; iter++) {
    //double t1 = MPI_Wtime();
    // W[0]
    T_C["ij*"] = V["ijk"] * W[2]["k*"];
    T_BC["*i"] = T_C["ij*"] * W[1]["j*"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];
    CC["ij"] = W[2]["ki"] * W[2]["kj"];

    S["ij"] = BB["ij"] * CC["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W0_trans = Matrix<>(rank, V.lens[0]);
    T_BC.solve_spd(S, W0_trans);
    W[0]["ij"] = W0_trans["ji"];

    // W[1]
    T_AC["*j"] = T_C["ij*"] * W[0]["i*"];
    AA["ij"] = W[0]["ki"] * W[0]["kj"];

    S["ij"] = AA["ij"] * CC["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W1_trans = Matrix<>(rank, V.lens[1]);
    T_AC.solve_spd(S, W1_trans);
    W[1]["ij"] = W1_trans["ji"];

    // W[2]
    T_A["jk*"] = V["ijk"] * W[0]["i*"];
    T_AB["*k"] = T_A["jk*"] * W[1]["j*"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];

    S["ij"] = AA["ij"] * BB["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W2_trans = Matrix<>(rank, V.lens[2]);
    T_AB.solve_spd(S, W2_trans);
    W[2]["ij"] = W2_trans["ji"];

    //double t2 = MPI_Wtime();
    //if (dw.rank == 0) {
    //  printf("dt iteration took %lf seconds\n", t2 - t1);
    //}

    int num_dw_break = 0;
    for (int i = 0; i < V.order; i++) {
      dW[i]["ij"] = W[i]["ij"] - W_prev[i]["ij"];
      W_prev[i]["ij"] = W[i]["ij"];
      double norm_dW = dW[i].norm2();
      double norm_W = W[i].norm2();
      if (abs(norm_dW / norm_W) < tol_init)
        num_dw_break++;
      else
        break;
    }
    if (num_dw_break == V.order) {
      iter++;
      delete[] W_prev;
      // tALS.end();
      return;
    }
    // print .
    if (iter % 10 == 0 && dw.rank == 0)
      printf(".");
    // print the gradient norm
    if ((iter % resprint == 0 || iter == maxiter) && (maxiter != 0)) {
      double st_time1 = MPI_Wtime();
      // diffnorm
      Matrix<> AA = Matrix<>(rank, rank);
      Matrix<> BB = Matrix<>(rank, rank);
      Matrix<> CC = Matrix<>(rank, rank);
      AA["ab"] = W[0]["ka"] * W[0]["kb"];
      BB["ab"] = W[1]["ka"] * W[1]["kb"];
      CC["ab"] = W[2]["ka"] * W[2]["kb"];
      double inner_W = AA["ab"] * BB["ab"] * CC["ab"];
      T_AB["*k"] = T_A["jk*"] * W[1]["j*"];
      double T_ABC = T_AB["ab"] * W[2]["ba"];
      diffnorm_V = sqrt(Vnorm * Vnorm + inner_W - 2. * T_ABC);
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      double fitness = 1. - diffnorm_V / Vnorm;
      if (dw.rank == 0) {
        cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter
             << "  [fitness]  " << fitness << "  [pp_update]  " << 0
             << "  [diffV]  " << diffnorm_V << "  [dtime]  " << dtime << "\n";
        Plot_File << V.lens[0] << "," << iter << "," << fitness << "," << 0
                  << "," << diffnorm_V << "," << dtime << "\n";
        if (iter % 100 == 0 && iter != 0) { // flush
          Plot_File << endl;
        }
      }
    }
  }
  // tALS.end();
}

void initialize_tree(Tensor<> &V, Matrix<> *W, Tensor<> &T_copy, Tensor<> &T_A0, Tensor<> &T_B0,
                     Tensor<> &T_C0, Matrix<> &T_A0B0, Matrix<> &T_B0C0,
                     Matrix<> &T_A0C0, int *order_Ta, int *order_Tb,
                     int *order_Tc, int partition, World &dw) {
  // Timer_epoch inittree("initialize_tree");
  // inittree.begin();
  // double t1 = MPI_Wtime();

  if (partition == 0) {
    T_A0 = Tensor<>(3, order_Ta, dw);
    T_B0 = Tensor<>(3, order_Tb, dw);
    T_C0 = Tensor<>(3, order_Tc, dw);
    T_A0["jka"] = V["ijk"] * W[0]["ia"];
    T_B0["ika"] = V["ijk"] * W[1]["ja"];
    T_C0["ija"] = V["ijk"] * W[2]["ka"];
  }
  else if (partition == 1) {
    if (dw.rank == 0) {
      cout << "partition is 1" << endl;
    }
    int np = dw.np;
    int syms[3] = {NS, NS, NS};
    CTF::Partition p(1, &np);
    T_A0 = Tensor<>(3, order_Ta, syms, dw, "ija", p["a"]);
    T_B0 = Tensor<>(3, order_Tb, syms, dw, "ija", p["a"]);
    T_C0 = Tensor<>(3, order_Tc, syms, dw, "ija", p["a"]);
    T_A0["jka"] = V["ijk"] * W[0]["ia"];
    T_B0["ika"] = V["ijk"] * W[1]["ja"];
    T_C0["ija"] = V["ijk"] * W[2]["ka"];
  }
  else if (partition == 2) {
    if (dw.rank == 0) {
      cout << "partition is 2" << endl;
    }
    int np = dw.np;
    int syms[3] = {NS, NS, NS};
    CTF::Partition p(1, &np);
    T_A0 = Tensor<>(3, order_Ta, syms, dw, "ija", p["a"]);
    T_B0 = Tensor<>(3, order_Tb, syms, dw, "ija", p["a"]);
    T_C0 = Tensor<>(3, order_Tc, syms, dw, "ija", p["a"]);

    // Tensor<> T_copy = Tensor<>(3, order_Ta, dw);

    // double t1 = MPI_Wtime();

    // T_copy["jka"] = V["ijk"] * W[0]["ia"];

    // double t2 = MPI_Wtime();

    // if (dw.rank == 0) {
    //   printf("pp initialization first step took %lf seconds\n", t2 - t1);
    // }

    T_A0["jka"] = T_copy["jka"];

    T_copy = Tensor<>(3, order_Tb, dw);
    T_copy["ika"] = V["ijk"] * W[1]["ja"];
    T_B0["ika"] = T_copy["ika"];

    T_copy = Tensor<>(3, order_Tc, dw);
    T_copy["ija"] = V["ijk"] * W[2]["ka"];
    T_C0["ija"] = T_copy["ija"];
  }

  T_A0B0["ka"] = T_A0["jka"] * W[1]["ja"];
  T_B0C0["ia"] = T_C0["ija"] * W[1]["ja"];
  T_A0C0["ja"] = T_A0["jka"] * W[2]["ka"];

  // double t2 = MPI_Wtime();
  // if (dw.rank == 0) {
  //   printf("pp initialization took %lf seconds\n", t2 - t1);
  // }
  // inittree.end();
}

void alscp_pp3_sub(Tensor<> &V, Matrix<> *W, Matrix<> *dW, Tensor<> &T_copy, double tol_init,
                   int maxiter, double &st_time, double lambda,
                   ofstream &Plot_File, int &iter, int resprint, int partition,
                   World &dw, double Vnorm) {
  // Timer_epoch tALS("alscp_pp3");
  // tALS.begin();

  Matrix<> *W_prev = new Matrix<>[V.order];
  for (int i = 0; i < V.order; i++) {
    W_prev[i] = Matrix<>(W[i].nrow, W[i].ncol);
    W_prev[i]["ij"] = W[i]["ij"];
    dW[i]["ij"] = 0.;
  }

  int rank = W[0].ncol;
  //  initialize matrix S
  Matrix<> S = Matrix<>(rank, rank);
  Matrix<> regul = Matrix<>(rank, rank);
  regul["ii"] = 1. * lambda;

  int dim1 = V.lens[0];
  int dim2 = V.lens[1];
  int dim3 = V.lens[2];

  Matrix<> AA = Matrix<>(rank, rank);
  Matrix<> BB = Matrix<>(rank, rank);
  Matrix<> CC = Matrix<>(rank, rank);

  Matrix<> T_BC = Matrix<>(rank, dim1);
  Matrix<> T_AC = Matrix<>(rank, dim2);
  Matrix<> T_AB = Matrix<>(rank, dim3);

  int order_Tc[3] = {dim1, dim2, rank};
  int order_Ta[3] = {dim2, dim3, rank};
  int order_Tb[3] = {dim1, dim3, rank};

  Tensor<> T_A0;
  Tensor<> T_B0;
  Tensor<> T_C0;

  Matrix<> T_B0C0 = Matrix<>(dim1, rank);
  Matrix<> T_A0C0 = Matrix<>(dim2, rank);
  Matrix<> T_A0B0 = Matrix<>(dim3, rank);

  initialize_tree(V, W, T_copy, T_A0, T_B0, T_C0, T_A0B0, T_B0C0, T_A0C0, order_Ta,
                  order_Tb, order_Tc, partition, dw);

  double diffnorm_V = 1000;
  int init_iter = iter;

  for (; iter <= maxiter; iter++) {
    //double t1 = MPI_Wtime();
    // W[0]
    T_BC["*i"] =
        T_B0C0["i*"] + T_C0["ij*"] * dW[1]["j*"] + T_B0["ik*"] * dW[2]["k*"];
    Matrix<> dBB = Matrix<>(rank, rank);
    Matrix<> dCC = Matrix<>(rank, rank);
    dBB["bc"] = dW[1]["ab"] * W[1]["ac"];
    dCC["bc"] = dW[2]["ab"] * W[2]["ac"];
    dBB["bc"] = dBB["bc"] * dCC["bc"];
    T_BC["ac"] = T_BC["ac"] + W[0]["cb"] * dBB["ab"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];
    CC["ij"] = W[2]["ki"] * W[2]["kj"];

    S["ij"] = BB["ij"] * CC["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W0_trans = Matrix<>(rank, V.lens[0]);
    T_BC.solve_spd(S, W0_trans);
    W[0]["ij"] = W0_trans["ji"];
    dW[0]["ij"] = W[0]["ij"] - W_prev[0]["ij"];

    // W[1]
    T_AC["*j"] =
        T_A0C0["j*"] + T_C0["ij*"] * dW[0]["i*"] + T_A0["jk*"] * dW[2]["k*"];
    Matrix<> dAA = Matrix<>(rank, rank);
    dAA["bc"] = dW[0]["ab"] * W[0]["ac"];
    dAA["bc"] = dAA["bc"] * dCC["bc"];
    T_AC["ac"] = T_AC["ac"] + W[1]["cb"] * dAA["ab"];

    AA["ij"] = W[0]["ki"] * W[0]["kj"];

    S["ij"] = AA["ij"] * CC["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W1_trans = Matrix<>(rank, V.lens[1]);
    T_AC.solve_spd(S, W1_trans);
    W[1]["ij"] = W1_trans["ji"];
    dW[1]["ij"] = W[1]["ij"] - W_prev[1]["ij"];

    // W[2]
    T_AB["*k"] =
        T_A0B0["k*"] + T_B0["ik*"] * dW[0]["i*"] + T_A0["jk*"] * dW[1]["j*"];
    dAA["bc"] = dW[0]["ab"] * W[0]["ac"];
    dBB["bc"] = dW[1]["ab"] * W[1]["ac"];
    dAA["bc"] = dAA["bc"] * dBB["bc"];
    T_AB["ac"] = T_AB["ac"] + W[2]["cb"] * dAA["ab"];

    BB["ij"] = W[1]["ki"] * W[1]["kj"];

    S["ij"] = AA["ij"] * BB["ij"];
    S["ij"] += regul["ij"];
    Matrix<> W2_trans = Matrix<>(rank, V.lens[2]);
    T_AB.solve_spd(S, W2_trans);
    W[2]["ij"] = W2_trans["ji"];
    dW[2]["ij"] = W[2]["ij"] - W_prev[2]["ij"];

    //double t2 = MPI_Wtime();

    //if (dw.rank == 0) {
    //  printf("pp middle step took %lf seconds\n", t2 - t1);
   // }

    for (int i = 0; i < V.order; i++) {
      double norm_dW = dW[i].norm2();
      double norm_W = W[i].norm2();
      if (abs(norm_dW / norm_W) > tol_init) {
        iter++;
        delete[] W_prev;
        // tALS.end();
        return;
      }
    }
    // print .
    if (iter % 10 == 0 && dw.rank == 0)
      printf(".");
    // print the gradient norm
    if ((iter % resprint == 0 || iter == maxiter || iter == init_iter) && (maxiter != 0)) {
      double st_time1 = MPI_Wtime();
      // diffnorm
      Matrix<> AA = Matrix<>(rank, rank);
      Matrix<> BB = Matrix<>(rank, rank);
      Matrix<> CC = Matrix<>(rank, rank);
      AA["ab"] = W[0]["ka"] * W[0]["kb"];
      BB["ab"] = W[1]["ka"] * W[1]["kb"];
      CC["ab"] = W[2]["ka"] * W[2]["kb"];
      double inner_W = AA["ab"] * BB["ab"] * CC["ab"];
      T_AB["*k"] =
          T_A0B0["k*"] + T_B0["ik*"] * dW[0]["i*"] + T_A0["jk*"] * dW[1]["j*"];
      dAA["bc"] = dW[0]["ab"] * W[0]["ac"];
      dBB["bc"] = dW[1]["ab"] * W[1]["ac"];
      dAA["bc"] = dAA["bc"] * dBB["bc"];
      T_AB["ac"] = T_AB["ac"] + W[2]["cb"] * dAA["ab"];
      double T_ABC = T_AB["ab"] * W[2]["ba"];
      diffnorm_V = sqrt(Vnorm * Vnorm + inner_W - 2. * T_ABC);
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      double fitness = 1. - diffnorm_V / Vnorm;
      if (dw.rank == 0) {
        cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter
             << "  [fitness]  " << fitness << "  [pp_update]  " << 1
             << "  [diffV]  " << diffnorm_V << "  [dtime]  " << dtime << "\n";
        Plot_File << V.lens[0] << "," << iter << "," << fitness << "," << 1
                  << "," << diffnorm_V << "," << dtime << "\n";
        if (iter % 100 == 0 && iter != 0) { // flush
          Plot_File << endl;
        }
      }
    }
  }
  // tALS.end();
}

bool alscp_pp3(Tensor<> &V, Matrix<> *W, int maxiter, double pp_res_tol,
               double lambda, ofstream &Plot_File, int resprint, int partition,
               World &dw, double Vnorm) {

  double st_time = MPI_Wtime();
  int iter = 0;
  double diffnorm_V = 1.;
  Tensor<> T_A0;
  // initialize dW
  Matrix<> *dW = new Matrix<>[V.order];
  for (int j = 0; j < V.order; j++) {
    dW[j] = Matrix<>(W[j].nrow, W[j].ncol);
    dW[j]["ij"] = 0.;
  }

  while (iter <= maxiter) {

    if (dw.rank == 0) {
      printf("DT starts from %d\n", iter);
    }

    alscp_dt3_sub(V, W, dW, T_A0, pp_res_tol, maxiter, st_time, lambda, Plot_File,
                  iter, resprint, dw, Vnorm);

    if (dw.rank == 0) {
      printf("pairwise perturbation starts from %d\n", iter);
    }

    alscp_pp3_sub(V, W, dW, T_A0, pp_res_tol, maxiter, st_time, lambda, Plot_File,
                  iter, resprint, partition, dw, Vnorm);
  }
  if (dw.rank == 0) {
    printf("tf took %lf seconds\n", MPI_Wtime() - st_time);
  }

  Plot_File.close();
  delete[] dW;
  if (iter == maxiter + 1)
    return false;
  else
    return true;
}

bool alscp_pp3_bench(Tensor<> &V, Matrix<> *W, int maxiter, double pp_res_tol,
                     double lambda, ofstream &Plot_File, int resprint,
                     int partition, World &dw, double Vnorm) {

  double st_time = MPI_Wtime();
  double diffnorm_V = 1.;
  Tensor<> T_A0;
  // initialize dW
  Matrix<> *dW = new Matrix<>[V.order];
  for (int j = 0; j < V.order; j++) {
    dW[j] = Matrix<>(W[j].nrow, W[j].ncol);
    dW[j]["ij"] = 0.;
  }

  for (int outer_iter = 0; outer_iter < maxiter; outer_iter++) {
    int iter = 0;

    if (dw.rank == 0) {
      printf("DT starts from %d\n", iter);
    }

    alscp_dt3_sub(V, W, dW, T_A0, pp_res_tol, 3, st_time, lambda, Plot_File, iter,
                  resprint, dw, Vnorm);

    if (dw.rank == 0) {
      printf("pairwise perturbation starts from %d\n", iter);
    }
    iter = 0;
    alscp_pp3_sub(V, W, dW, T_A0, pp_res_tol, 3, st_time, lambda, Plot_File, iter,
                  resprint, partition, dw, Vnorm);
  }
  if (dw.rank == 0) {
    printf("td took %lf seconds\n", MPI_Wtime() - st_time);
  }

  Plot_File.close();
  delete[] dW;
  return true;
}
