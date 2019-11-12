
#include "../common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int size_, int r, World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {
  optimizer = new Optimizer(order, r, dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int size_, int r, int update_rank,
                           World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {
  // NOTE: this constructor is only for the optimizer with low rank updates
  optimizer = new Optimizer(order, r, update_rank, dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int size_, int r, int update_rank,
                           int randomsvd, World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {
  // NOTE: this constructor is only for the optimizer with low rank updates
  optimizer = new Optimizer(order, r, update_rank, randomsvd, dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int *size_, int *r, World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {

  int size0 = size_[0];
  int rank0 = r[0];
  for (int i = 1; i < order; i++) {
    assert(this->size[i] == size0);
    assert(this->rank[i] == rank0);
  }

  optimizer = new Optimizer(order, r[0], dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::Init(Tensor<dtype> *input, Matrix<dtype> *mat,
                                 double lambda) {

  Decomposition<dtype>::Init(input, mat);
  World *dw = this->world;
  // initialize grad_W
  if (grad_W != NULL) {
    delete[] grad_W;
  }
  grad_W = new Matrix<>[this->order];
  for (int i = 0; i < this->order; i++) {
    grad_W[i] = Matrix<dtype>(this->size[i], this->rank[i], *dw);
    grad_W[i].fill_random(0, 1);
  }
  // configure the optimizer
  this->optimizer->configure(input, mat, grad_W, lambda);
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::print_grad(int i) const {
  assert(grad_W != NULL);
  grad_W[i].print();
}

template <typename dtype, class Optimizer> CPD<dtype, Optimizer>::~CPD() {
  if (grad_W != NULL) {
    delete[] grad_W;
  }
  if (optimizer != NULL) {
    delete optimizer;
  }
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::update_gradnorm() {
  gradnorm = 0;
  for (int i = 0; i < this->order; i++) {
    gradnorm += this->grad_W[i].norm2() * this->grad_W[i].norm2();
  }
  gradnorm = sqrt(gradnorm);
}

template <typename dtype, class Optimizer>
bool CPD<dtype, Optimizer>::als(double tol, double timelimit, int maxsweep,
                                int resprint, ofstream &Plot_File, bool bench) {

  cout.precision(13);

  World *dw = this->world;
  double st_time = MPI_Wtime();
  int iters = 0;
  double sweeps = 0;
  double diffnorm_V = 1000.;

  if (bench == false) {
    if (dw->rank == 0)
      Plot_File << "[dim],[iter],[gradnorm],[tol],[pp_update],[diffV],[dtime]"
                << "\n"; // Headings for file
  }

  while (int(sweeps) <= maxsweep) {
    // print the gradient norm
    if (iters % resprint == 0 || sweeps >= maxsweep || sweeps == 0) {
      double st_time1 = MPI_Wtime();
      update_gradnorm();
      // residual
      Tensor<dtype> V_build;
      build_V(V_build, this->W, this->order, *dw);
      Tensor<dtype> diff_V = *(this->V);
      diff_V[seq_V] = (*this->V)[seq_V] - V_build[seq_V];
      diffnorm_V = diff_V.norm2();
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      if (bench == false) {
        if (dw->rank == 0) {
          cout << "  [dim]=  " << (this->V)->lens[0]
               << "  [sweeps]=  " << sweeps << "  [gradnorm]  " << gradnorm
               << "  [tol]  " << tol << "  [pp_update]  " << 0
               << "  [residual]  " << diffnorm_V << "  [dtime]  " << dtime
               << "\n";
          Plot_File << (this->V)->lens[0] << "," << sweeps << "," << gradnorm
                    << "," << tol << "," << 0 << "," << diffnorm_V << ","
                    << dtime << "\n";
          // flush the contents to csv
          if (iters % 100 == 0 && iters != 0) {
            Plot_File << endl;
          }
        }
      } else {
        if (dw->rank == 0 && int(iters) != 0) {
          cout << "  [dimension tree step time]  " << dtime << "\n";
          Plot_File << "[DTtime]"
                    << "," << dtime << "\n";
        }
      }
      if ((gradnorm < tol) || MPI_Wtime() - st_time > timelimit)
        break;
    }

    sweeps += this->optimizer->step();
    iters += 1;

    // Normalize(this->W, this->order, *dw);
    // print .
    if (iters % 10 == 0 && dw->rank == 0)
      printf(".");
  }
  if (dw->rank == 0) {
    printf("\nIters = %d Final proj-grad norm %E \n", iters, gradnorm);
    printf("tf took %lf seconds\n", MPI_Wtime() - st_time);
  }
  if (bench == false) {
    Plot_File.close();
  }
  if (sweeps == maxsweep + 1)
    return false;
  else
    return true;
}
