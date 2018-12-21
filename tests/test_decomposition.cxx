
#include "../src/decomposition.h"
#include "../src/CP.h"
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
	CPD<double> decom(3,5,2,dw);
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
	decom.print_grad(0);
	decom.print_grad(1);

	// test als
	decom.als(1e-5, 1000, 1000);
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

	TEST_decomposition(dw);
	TEST_CPD(dw);


	MPI_Finalize();

}
/**
 * @} 
 * @}
 */


// #endif


