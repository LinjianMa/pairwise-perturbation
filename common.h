#ifndef __COMMON_ALS_H__
#define __COMMON_ALS_H__

#include <ctf.hpp>
using namespace CTF;

void vec2str(vector<int> vec, string &seq_out);

void build_V(Tensor<> &V, Matrix<> *W, int order, World &dw);

void mttkrp_map_DT(map<string, Tensor<>> &mttkrp_map,
                   map<string, string> &parent, map<string, string> &sibling,
                   Tensor<> &V, Matrix<> *W, string args, World &dw);

void swap_char(char *seq, int i, int j);

Matrix<> unroll_tensor_contraction(Tensor<> &T, int i);

void Construct_Dimension_Tree(map<string, string> &parent,
                              map<string, string> &sibling, int start, int end);

// common functions
void unit_tensor(Tensor<> &V, int N, int s, World &dw);

void Gram_Schmidt(Vector<> &A, Vector<> &B);

Vector<> **Gen_vector_condition(int *lens, int dim, int R, double condition);

Tensor<> Gen_tensor_condition(int *lens, int dim, int R, int base,
                              double condition, World &dw);

/**
 * \brief Identity tensor: I x I x I x ...
 */
Tensor<> identitiy_tensor(int N, int s, World &dw);

/**
 * \brief laplacian tensor:
 * 3d example : I x D x I + D x I x I + I x I x D
 */
void random_laplacian_tensor(Tensor<> &V, int N, int s, bool sparse_V,
                             World &dw);

/**
 * \brief laplacian tensor:
 * 3d example : I x D x I + D x I x I + I x I x D
 */
void laplacian_tensor(Tensor<> &V, int N, int s, bool sparse_V, World &dw);

void Normalize(Matrix<> *W, int N, World &dw);

void SVD_solve(Matrix<> &M, Matrix<> &W, Matrix<> &S);

void cholesky_solve(Matrix<> &M, Matrix<> &W, Matrix<> &S);

void SVD_solve_mod(Matrix<> &M, Matrix<> &W, Matrix<> &W_init, Matrix<> &dW,
                   Matrix<> &S, double ratio_step);

void matrixDot(Matrix<> &result, Matrix<> &matrix1, Matrix<> &matrix2);

void get_rankR_update_svd(int R, Matrix<> &U, Vector<> &sigma, Matrix<> &VT,
                          Matrix<> &M, Matrix<> &A, Matrix<> &gamma,
                          bool random);

void get_rankR_update_cholesky(int R, Matrix<> &U, Vector<> &sigma,
                               Matrix<> &VT, Matrix<> &M, Matrix<> &A,
                               Matrix<> &gamma, bool random);

void apply_rankR_update(Matrix<> &U, Vector<> &sigma, Matrix<> &VT, Matrix<> &A,
                        Tensor<> &V, Tensor<> *cached_tensor, int mode,
                        World &dw);

// Gauss-Seidel relaxation for A*Gamma = F
void Gauss_Seidel(Matrix<> &A, Matrix<> &F, Matrix<> &Gamma, int maxits);

void fold_unfold(Tensor<> &X, Tensor<> &Y);

/**
 * \brief To calculate the Khatri-Rao Product of W[i]
 *  H_T: output solution
 *  W[i]: input matrix
 *  index: sequence for W[i] to be used
 *  lens_H: lens of each dimension in H_T
 */
void KhatriRaoProduct(Tensor<> &H_T, Matrix<> *W, int *index, int *lens_H,
                      World &dw);

/**
 * \brief To calculate the Khatri-Rao Product of W[i] and contract with V
 *  M: output solution
 *  V: input tensor
 *  W[i]: input matrixs
 *  index: sequence for W[i] to be used
 *  lens_H: lens of each dimension in H_T
 *  M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
 */
void KhatriRao_contract(Matrix<> &M, Tensor<> &V, Matrix<> *W, int *index,
                        int *lens_H, World &dw);

/**
 *  \brief subproblem grad_W[i]
 */
void gradsubprob(Matrix<> &M, Matrix<> &S, Matrix<> &W, Matrix<> &grad_W);

/**
 * \brief initialize grad_W
 */
void gradient_CP(Tensor<> &V, Matrix<> *W, Matrix<> *grad_W, World &dw);

void char_string_copy(char *a, int start_a, string &b, int start_b, int len);

Tensor<> Gen_collinearity(int *lens, int dim, int R, double col_min,
                          double col_max, World &dw);

void print_str(char *str);

#endif
