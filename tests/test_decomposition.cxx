
#include "../src/decomposition.h"
#include "../src/CP.h"
#include "../src/optimizer/cp_als_optimizer.h"
#include "../src/optimizer/cp_simple_optimizer.h"
#include "../src/optimizer/cp_dt_optimizer.h"
#include "../src/optimizer/cp_msdt_optimizer.h"


#include <ctf.hpp>


using namespace CTF;

void TEST_decomposition(World & dw) {

    // test dimension
    Decomposition<double> decom(3,5,2,dw);
    cout << decom.order << endl;
    assert(decom.order == 3);
    assert(decom.rank[0] == 2);

    // test init
    int lens[3];
    for (int i=0; i<3; i++) lens[i]=5;
    Tensor<> *V = new Tensor<>(3,lens,dw);
    V->fill_random(0,1);
    Matrix<> *W = new Matrix<>[3];
    for (int i=0; i<3; i++) {
        W[i] = Matrix<>(5,2,dw);
        W[i].fill_random(0,1); 
    }
    decom.Init(V,W);
    decom.print_W(0);
    decom.print_W(1);   
}

void TEST_CPD(World & dw) {

    // test dimension
    CPD<double, CPDTOptimizer<double>> decom(6,13,5,dw);
    cout << decom.order << endl;
    assert(decom.order == 6);
    assert(decom.rank[0] == 5);

    // test init
    int lens[6];
    for (int i=0; i<6; i++) lens[i]=13;
    Tensor<> *V = new Tensor<>(6,lens,dw);
    V->fill_random(0,1);
    Matrix<> *W = new Matrix<>[6];
    for (int i=0; i<6; i++) {
        W[i] = Matrix<>(13,5,dw);
        W[i].fill_random(0,1); 
    }
    decom.Init(V,W);
    // decom.print_W(0);
    // decom.print_W(1);    
    // decom.print_grad(0);
    // decom.print_grad(1);

    ofstream Plot_File("results/test.csv"); 

    // test als
    decom.als(1e-5, 1000, 30, 100, Plot_File);
}

// #ifndef TEST_SUITE
/**
 * \brief Forms N-by-N DFT matrix A and inverse-dft iA and checks A*iA=I
 */
int main(int argc, char ** argv){
    int logn;
    int64_t n;

    MPI_Init(&argc, &argv);

    World dw(argc, argv);

    // TEST_decomposition(dw);
    TEST_CPD(dw);


    MPI_Finalize();

}
/**
 * @} 
 * @}
 */


// #endif


