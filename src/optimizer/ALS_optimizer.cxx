
// #include "decomposition.h"
#include <ctf.hpp>

using namespace CTF;

template<typename dtype>  
Optimizer<dtype>::Optimizer(int order, int r, World & dw) {

	this->world = & dw; 
	this->order = order;
	this->rank = r;

	// S
	S = Matrix<>(r,r);

}

template<typename dtype>  
Optimizer<dtype>::~Optimizer(){
}

template<typename dtype>
void Optimizer<dtype>::configure(Tensor<dtype>* input, Matrix<dtype>* mat, Matrix<dtype>* grad, double lambda){

	assert(input->order == order);
	cout << order << endl;

	for (int i=0; i< order; i++) {
		assert(mat[i].ncol == rank);
	}

	if (V != NULL) {
		delete V;
	}
	if (W != NULL) {
		delete[] this->W;
	}
	if (grad_W != NULL) {
		delete[] this->grad_W;
	}
	this->V = input;
	this->W = mat;
	this->grad_W = grad;

    regul =Matrix<dtype>(mat[0].ncol,mat[0].ncol);
    regul["ii"] =  1.*lambda;
}

