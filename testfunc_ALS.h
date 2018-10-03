#ifndef __TEST_ALS__
#define __TEST_ALS__

#include <ctf.hpp>
using namespace CTF;

/**
 * \brief CP decomposition of random tensor using simple ALS
 */
void TEST_alsCP(int dim, 
        int * lens, 
        int K, 
        World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsCP(int N,       // Dimension of the tensor
                 int s,       // size in each dimension
                 int K,       // Decomposition rank
                 bool sparse_V,   // Whether V is set to be sparse or not
                 World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsCP_DT(int N,        // Dimension of the tensor
                  int s,        // size in each dimension
                  int K,      // Decomposition rank
                  bool sparse_V,    // Whether V is set to be sparse or not
                  World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsCP_mod(int N,       // Dimension of the tensor
                   int s,       // size in each dimension
                   int K,       // Decomposition rank
                   bool sparse_V,   // Whether V is set to be sparse or not
                   World & dw);

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsCP(int s, 
                int K, 
                World & dw);

/**
 * \brief test the identity tensor
 */
void TEST_identity_tensor(int N, int s, World & dw);

/**
 * \brief test the svd solver
 */
void TEST_SVD_solve(int N, World & dw);

/**
 * \brief test the correctness of Laplacian tensor
 */
void TEST_laplacian_tensor(int N, 
               int s, 
               bool sparse_V,
               World & dw);

/**
 * \brief test the Gauss Seidel method
 *        Gauss-Seidel relaxation for A*Gamma = F
 */
void TEST_gauss_seidel(int N,
             int K, 
             World & dw);


/**
 * \brief test the Hosvd method
 */
void TEST_hosvd(int N,
        int * T_lens,
        int * ranks,
        World & dw);

/**
 * \brief test the Hosvd method
 */
void TEST_alsTucker(int N,
          int * V_lens,
          int * ranks,
          World & dw);

/**
 * \brief test the Hosvd method
 */
void TEST_alsTucker_DT(int N,
             int * V_lens,
             int * ranks,
             World & dw);

void TEST_Gram_Schmidt();

void TEST_Gen_vector_condition(int * lens,
                  int dim,
                  int R,
                  double condition);

void TEST_Gen_tensor_condition(int * lens,
                   int dim,
                   int R,       // generate how many independent vectors in each mode
                   int base,    // how many basis
                   int K,
                   double condition, 
                   World & dw) ;

void TEST_Gen_tensor_condition_pp(int * lens,
                    int dim,
                    int R,       // generate how many independent vectors in each mode
                    int base,    // how many basis
                    int K,
                    double condition, 
                    World & dw) ;

void TEST_unit_tensor_pp(int* lens,
             int dim,
             int K,
             double condition, 
             World & dw) ;

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker(int N,       // Dimension of the tensor
                   int s,       // size in each dimension
                   int K,       // Decomposition rank
                   bool sparse_V,   // Whether V is set to be sparse or not
                   double criteria,    // global stopping criteria
                   ofstream & Plot_File, 
                   World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_sparse_laplacian_alsTucker_PP(int N,        // Dimension of the tensor
                    int s,        // size in each dimension
                    int K,      // Decomposition rank
                    bool sparse_V,    // Whether V is set to be sparse or not
                    double criteria,    // global stopping criteria
                    double tol_init,
                    ofstream & Plot_File,
                    World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_laplacian_alsTucker(int N,       // Dimension of the tensor
                   int s,       // size in each dimension
                   int K,       // Decomposition rank
                   bool sparse_V,   // Whether V is set to be sparse or not
                   double criteria,    // global stopping criteria
                   ofstream & Plot_File, 
                   World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_laplacian_alsTucker_PP(int N,        // Dimension of the tensor
                    int s,        // size in each dimension
                    int K,      // Decomposition rank
                    bool sparse_V,    // Whether V is set to be sparse or not
                    double criteria,    // global stopping criteria
                    double tol_init,
                    ofstream & Plot_File,
                    World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_alsTucker(int N,       // Dimension of the tensor
                   int s,       // size in each dimension
                   int K,       // Decomposition rank
                   bool sparse_V,   // Whether V is set to be sparse or not
                   double criteria,    // global stopping criteria
                   ofstream & Plot_File, 
                   World & dw);

/**
 * \brief CP decomposition of laplacian tensor using simple ALS
 */
void TEST_random_alsTucker_PP(int N,        // Dimension of the tensor
                    int s,        // size in each dimension
                    int K,      // Decomposition rank
                    bool sparse_V,    // Whether V is set to be sparse or not
                    double criteria,    // global stopping criteria
                    double tol_init,
                    ofstream & Plot_File,
                    World & dw);

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsTucker(int s, 
                    int K, 
                  bool sparse_V,    // Whether V is set to be sparse or not
                  double criteria,    // global stopping criteria
                  ofstream & Plot_File,
                  World & dw);

/**
 * \brief CP decomposition of dense tensor using simple als
 *        This test is not functional now
 */
void TEST_dense_uniform_alsTucker_PP(int s, 
                     int K, 
                   bool sparse_V,   // Whether V is set to be sparse or not
                   double criteria,    // global stopping criteria
                   double tol_init,
                   ofstream & Plot_File,
                   World & dw);

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_3d_poisson_Tucker(int N,        // Dimension of the tensor
              int s,        // size in each dimension
              int K,      // Decomposition rank
              bool sparse_V,    // Whether V is set to be sparse or not
              double criteria,
              ofstream & Plot_File,
              World & dw);

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_construct_Tucker(int N,       // Dimension of the tensor
               int s,       // size in each dimension
               int K,       // Decomposition rank
               bool sparse_V,   // Whether V is set to be sparse or not
               double criteria,
               ofstream & Plot_File,
               World & dw);

/**
 * \brief Tucker decomposition of laplacian tensor using simple ALS
 */
void TEST_construct_Tucker_pp(int N,        // Dimension of the tensor
               int s,       // size in each dimension
               int K,       // Decomposition rank
               bool sparse_V,   // Whether V is set to be sparse or not
               double criteria,
               double tol_init,
               ofstream & Plot_File,
               World & dw);


#endif