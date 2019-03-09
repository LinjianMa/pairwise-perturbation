
// #include "decomposition.h"
#include <ctf.hpp>
#include "../../common.h"


using namespace CTF;

template<typename dtype>  
DTOptimizer<dtype>::DTOptimizer(int order, int r, World & dw)
	: Optimizer<dtype>(order, r, dw){

	// make the char seq_V and seq
	seq[order] = '\0'; 
	seq_V[order] = '\0'; 
	for (int j=0; j<order; j++) {
		seq[j] = 'a'+j;
		seq_V[j] = seq[j];
	}

	// construct the tree
	Construct_Dimension_Tree(parent, sibling, 0, order-1);

}

template<typename dtype>  
DTOptimizer<dtype>::~DTOptimizer(){
	// delete S;
}

template<typename dtype>
void DTOptimizer<dtype>::step() {

	World * dw = this->world;
	int order = this->order; 

	// clear the Hash Table
	mttkrp_map.clear();
	// iteration on W[i]
	for (int i=0; i<order; i++) { 
		//make the char
		swap_char(seq_V, i, order-1);
		/*  construct Matrix M
		*	M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
		*/
		int lens_H[order];
		int index[order];
		for (int j=0; j<order-1; j++) {
			index[j] = (int)(seq_V[j]-'a');
			lens_H[j] = (this->V)->lens[index[j]];
		}
		index[order-1] = (int)(seq_V[order-1]-'a');
		lens_H[order-1] = (this->W)[i].ncol;
		/* initialize matrix M
		*/
		// make args
		char args[2];
		args[1] = '\0';
		args[0] = i+'a';
		if (mttkrp_map.find(parent[args])==mttkrp_map.end()) {
			mttkrp_map_DT(mttkrp_map, parent, sibling, *(this->V) , this->W, parent[args], *dw);
		}
		Matrix<> M = Matrix<>(this->W[i].nrow,this->W[i].ncol);
		if (sibling[args].length()==1) {
			char seq[3],seq_A[3],seq_p[4];
			seq[2] = '\0'; seq_A[2] = '\0'; seq_p[3] = '\0';
			seq[1] = '*'; seq_A[1] = '*'; seq_p[2] = '*';
			seq[0] = args[0]; seq_p[0] = parent[args][0]; seq_p[1] = parent[args][1];
			if (seq_p[0]==seq[0]) seq_A[0] = seq_p[1];
			else seq_A[0] = seq_p[0];
			M[seq] = mttkrp_map[parent[args]][seq_p]*this->W[int(seq_A[0]-'a')][seq_A];
		} else {
			char seq[3],seq_A1[3],seq_A2[3],seq_p[5];
			seq[2] = '\0'; seq_A1[2] = '\0'; seq_A2[2] = '\0'; seq_p[4] = '\0';
			seq[1] = '*'; seq_A1[1] = '*'; seq_A2[1] = '*'; seq_p[3] = '*';
			seq[0] = args[0]; seq_p[0] = parent[args][0]; seq_p[1] = parent[args][1]; seq_p[2] = parent[args][2];
			if (seq_p[0]==seq[0]) {
				seq_A1[0] = seq_p[1];
				seq_A2[0] = seq_p[2];
			}
			else {
				seq_A1[0] = seq_p[0];
				seq_A2[0] = seq_p[1];
			}
			M[seq] = mttkrp_map[parent[args]][seq_p]*this->W[int(seq_A1[0]-'a')][seq_A1]*this->W[int(seq_A2[0]-'a')][seq_A2];				
		}
		// Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
		// KhatriRao_contract(M2, V, W, index, lens_H, dw);
		// calculating S
		this->S["ij"] = this->W[index[0]]["ki"]*this->W[index[0]]["kj"];
		for (int ii=1; ii<order-1; ii++) {
			this->S["ij"] = this->S["ij"]*(this->W[index[ii]]["ki"]*this->W[index[ii]]["kj"]);
		}
		this->S["ij"] += this->regul["ij"]; 
		// calculate gradient
		this->grad_W[i]["ij"] = -M["ij"]+this->W[i]["ik"]*this->S["kj"]; 
		SVD_solve(M, this->W[i], this->S);
		// recover the char
		swap_char(seq_V, i, order-1);
	}
}

