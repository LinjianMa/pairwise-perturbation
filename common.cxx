/** \addtogroup examples 
  * @{ 
  * \defgroup helper functions for multigrid
  * @{ 
  * \brief NTF algorithms based on projected gradient methods
  */
#include <ctf.hpp>
using namespace CTF;
//#define ERR_REPORT

/**
 * \brief Identity tensor: I x I x I x ...
 */
Tensor<> identitiy_tensor(int N, 
						  int s, 
						  World & dw) {
	int d = N/2;
	Matrix<> ident = Matrix<>(s,s,SP,dw);
	ident["ii"] = 1.;

	int *lens = new int[N];
	for (int i=0; i<N; i++) lens[i]=s;		
	Tensor<> I(N,true,lens,dw);

	Tensor<> * I_temp = new Tensor<>;
	(*I_temp) = ident;
	for (int i=1; i<d; i++) {
		Tensor<> I_temp2 = (*I_temp);
		// lens
		int *lens_temp = new int[2*i+2];
		for (int jj=0; jj<2*i+2; jj++) lens_temp[jj]=s;
		// I_temp		
		I_temp = new Tensor<>(2*i+2,true,lens_temp,dw);
		//build char
		char seq_I2[2*i+1]; seq_I2[2*i] = '\0';
		char seq_I1[2*i+3]; seq_I1[2*i+2] = '\0';
		for (int jj=0; jj<(2*i+2); jj++) seq_I1[jj] = 'a'+jj;
		for (int jj=2; jj<(2*i+2); jj++) seq_I2[jj-2] = 'a'+jj;
		(*I_temp)[seq_I1] = I_temp2[seq_I2]*ident["ab"];
	}
	I = (*I_temp);
    return I;
}

/**
 * \brief laplacian tensor: 
 * 3d example : I x D x I + D x I x I + I x I x D
 */
void laplacian_tensor(Tensor<>& V,
					  int N, 
					  int s, 
					  bool sparse_V,
					  World & dw){

	int d = N/2;
	// build D matrix
	Matrix<> D = Matrix<>(s,s,SP,dw);
	int64_t my_tot_nnz = s-1;
	int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
	double * vals = (double*)malloc(sizeof(double)*my_tot_nnz);
	for (int64_t row=0; row<my_tot_nnz; row++){
		inds[row] = row*s+row+1;
		if(dw.rank==0) vals[row] = -1.;
		else vals[row] = 0.;
	}
	D.write(my_tot_nnz, inds, vals);
	free(inds);
	free(vals);
	D["ij"] += D["ji"];
	D["ii"] = 2.;
	// build char for seq
	char seq[N+1]; seq[N] = '\0';
	for (int jj=0; jj<N; jj++) seq[jj] = 'a'+jj;
	/* k=1 */
	// initialize		
	Tensor<> I2 = identitiy_tensor(N-2, s, dw);
	// build char
	char seq_D[3] = "ab";
	char seq_I2[N-1]; seq_I2[N-2] = '\0';
	for (int jj=2; jj<N; jj++) seq_I2[jj-2] = 'a'+jj;
	// contract
	V[seq] += D[seq_D]*I2[seq_I2];
	// k=d
	// initialize		
	Tensor<> I1 = identitiy_tensor(N-2, s, dw);
	// build char
	char seq_I1[N-1]; seq_I1[N-2] = '\0';
	for (int jj=0; jj<N-2; jj++) seq_I1[jj] = 'a'+jj;
	for (int jj=N-2; jj<N; jj++) seq_D[jj-(N-2)] = 'a'+jj;
	// contract
	V[seq] += I1[seq_I1]*D[seq_D];
	// k in [2,d-1]
	for (int k=2; k<=d-1; k++) {
		Tensor<> I1 = identitiy_tensor(2*(k-1), s, dw);
		Tensor<> I2 = identitiy_tensor(2*(d-k), s, dw);
		// build char
		char seq_I1[2*(k-1)+1]; seq_I1[2*(k-1)] = '\0';
		char seq_I2[2*(d-k)+1]; seq_I2[2*(d-k)] = '\0';
		for (int jj=0; jj<2*(k-1); jj++) seq_I1[jj] = 'a'+jj;
		for (int jj=2*(k-1); jj<2*k; jj++) seq_D[jj-2*(k-1)] = 'a'+jj;
		for (int jj=2*k; jj<2*d; jj++) seq_I2[jj-2*k] = 'a'+jj;
		// contract
		V[seq] += I1[seq_I1]*D[seq_D]*I2[seq_I2];
	} 
}

void Normalize(Matrix<>* W, 
			   int N, 
			   World & dw) {
/*
	int R = W[0].ncol;
	double norm[N][R];
	double norm_sum[R];
	for (int i=0; i<R; i++) {
		norm_sum[i] = 1.;
		Matrix<> transform(R,1,dw);
		int64_t inds_t[1];
		double vals_t[1];
		inds_t[0] = i;
		if(dw.rank==0) vals_t[0] = 1.;
		else vals_t[0] = 0;
		transform.write(1,inds_t,vals_t);
		for (int j=0; j<N; j++) {
			Matrix<> W_part(W[j].nrow,1,dw);
			W_part["ij"] = W[j]["ik"]*transform["kj"];
			norm[j][i] = W_part.norm2();
			norm_sum[i] *= norm[j][i];
		}
		norm_sum[i] = pow(norm_sum[i],1./N);
	}
	// update the W
	for (int j=0; j<N; j++) {
		Matrix<> transform(R,R,dw);
		int64_t inds_t[R];
		double vals_t[R];
		for (int jj=0; jj<R; jj++) {
			inds_t[jj] = jj*R+jj;
			if(dw.rank==0) vals_t[jj] = norm_sum[jj]/norm[j][jj];
			else vals_t[0] = 0;	
		}
		transform.write(R,inds_t,vals_t);
		W[j]["ij"] = W[j]["ik"]*transform["kj"];
	}
*/
	double norm = 1;
	for (int i=0; i<N; i++) {
		norm = norm*W[i].norm2();
	}
	norm = pow(norm,1.0/N);
	for (int i=0; i<N; i++) {
		double norm_Wi = W[i].norm2();
		W[i]["ij"] = norm/norm_Wi*W[i]["ij"];
	}	
}

void SVD_solve(Matrix<>& M, 
			   Matrix<>& W, 
			   Matrix<>& S) {
	// Perform SVD
	Matrix<> U,VT;
	Vector<> s;
	S.svd(U,s,VT,S.ncol);
	Matrix<> S_reverse(S);
 	// reverse
 	Transform<> inv([](double & d){ d=1./d; });
	inv(s["i"]);
	S_reverse["ij"] = VT["ki"]*s["k"]*U["jk"];
	W["ij"] = M["ik"]*S_reverse["kj"];
}

void SVD_solve_mod(Matrix<>& M, 
				   Matrix<>& W,
				   Matrix<>& W_init,
				   Matrix<>& dW, 
				   Matrix<>& S) {
	// Perform SVD
	Matrix<> U,VT;
	Vector<> s;
	S.svd(U,s,VT,S.ncol);
	Matrix<> S_reverse(S);
 	// reverse
 	Transform<> inv([](double & d){ d=1./d; });
	inv(s["i"]);
	S_reverse["ij"] = VT["ki"]*s["k"]*U["jk"];
	W["ij"] = M["ik"]*S_reverse["kj"];
	dW["ij"] = W["ij"]-W_init["ij"];
}

// Gauss-Seidel relaxation for A*Gamma = F
void Gauss_Seidel(Matrix<>& A, 
				  Matrix<>& F,
				  Matrix<>& Gamma,
				  int maxits) {
	// extract lower triangular part of Gamma into nonsymmetric matrix
	// gives a directed adjacency matrix P
	Matrix<> Gamma_SH(Gamma.nrow,Gamma.ncol,SH);
	Gamma_SH["ij"] = Gamma["ij"];
	Gamma_SH["ij"] = 0.5*Gamma_SH["ij"];
	int nosym[] = {NS, NS};
  	Tensor<> G_Ut(Gamma_SH,nosym);
  	Matrix<> G_U(G_Ut);
  	Matrix<> G_L(Gamma);
  	G_L["ij"] = Gamma["ij"]-G_U["ij"];
	// reverse G_L
	Matrix<> U,VT;
	Vector<> p;
	G_L.svd(U,p,VT,G_L.ncol);
	Matrix<> GL_reverse(G_L);
 	// reverse
 	Transform<> inv([](double & d){ d=1./d; });
	inv(p["i"]);
	GL_reverse["ij"] = VT["ki"]*p["k"]*U["jk"];	
	// iteration
	for(int i=0; i<maxits; i++) {
		// A = A+(P\(F-A*Gamma)T)T;
		Matrix<> M(F.nrow,F.ncol);
		M["ij"] = F["ij"]-A["ik"]*Gamma["kj"];
		A["ij"] = A["ij"]+M["ik"]*GL_reverse["jk"];
		//A["ji"] = GL_reverse["ik"]*(F["jk"]-G_U["kl"]*A["jl"]);
	}
}

void fold_unfold(Tensor<>& X, Tensor<>& Y){
	int64_t * inds_X;
	double * vals_X;
	int64_t n_X;
	//if global index ordering is preserved between the two tensors, we can fold simply
	X.read_local(&n_X, &inds_X, &vals_X);
	Y.write(n_X, inds_X, vals_X);
}

/**
 * \brief To calculate the Khatri-Rao Product of W[i]
 *  H_T: output solution
 *  W[i]: input matrix
 *  index: sequence for W[i] to be used 
 *  lens_H: lens of each dimension in H_T
 */
void KhatriRaoProduct(Tensor<> & H_T, 
					  Matrix<> * W, 
					  int * index, 
					  int * lens_H, 
					  World & dw) {

	int K = H_T.lens[H_T.order-1];
	Tensor<> H_front = W[index[0]]; 
	Tensor<> H_temp; 
	for (int j=1; j<H_T.order-1; j++) {     // iterate on [ab]
		// make the char
		char seq[H_front.order+1], seq_f[H_front.order+2];
		seq[H_front.order] = '\0';
		seq_f[H_front.order+1] = '\0';
		for (int jj=0; jj<(H_front.order-1); jj++) {
			seq[jj] = 'a'+jj;
			seq_f[jj] = 'a'+jj;
		}
		seq[H_front.order-1] = 'k';
		seq_f[H_front.order-1] = 'j';
		seq_f[H_front.order] = 'k';
		// build len for H_temp
		int lens_W[j+2];
		for (int m=0; m<j+1; m++) {
			lens_W[m] = lens_H[m];
		}
		lens_W[j+1] = K;
		H_temp = Tensor<>(j+2, lens_W, dw);
		// contraction
		H_temp[seq_f] = H_front[seq]*W[index[j]]["jk"];
		H_front = H_temp;
	}
	H_T = H_front;
	return;
}

/**
 * \brief To calculate the Khatri-Rao Product of W[i] and contract with V
 *  M: output solution
 *  V: input tensor
 *  W[i]: input matrixs
 *  index: sequence for W[i] to be used 
 *  lens_H: lens of each dimension in H_T
 *	M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
 */
void KhatriRao_contract(Matrix<> & M, 
						Tensor<> & V, 
						Matrix<> * W, 
						int * index, 
						int * lens_H, 
						World &dw) {

	int K = W[0].ncol;
	Tensor<> V_front = V; 
	Tensor<> V_temp; 
	/* initial condition
	*/
	char seq[V_front.order+1], seq_f[V_front.order+1], seq_w[3];
	seq[V_front.order] = '\0';
	seq_f[V_front.order] = '\0';
	seq_w[2] = '\0';
	// make seq_w
	seq_w[0] = 'a'+index[0];
	seq_w[1] = 'k';
	// make seq
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
	}
	// make seq_f
	for (int jj=1; jj<V.order; jj++) {
		seq_f[jj-1] = 'a'+index[jj];
	}
	seq_f[V.order-1] = 'k';
	// build len for V_temp
	int lens_V[V.order];
	for (int m=0; m<V.order-1; m++) {
		lens_V[m] = lens_H[m];
	}
	lens_V[V.order-1] = K;
	V_temp = Tensor<>(V.order, lens_V, dw);
	// contraction
	V_temp[seq_f] = V_front[seq]*W[index[0]][seq_w];
	V_front = V_temp;
	/* loops
	*/
	for (int j=1; j<V.order-1; j++) {     // iterate on [ab]
		// make seq
		char seq[V_front.order+1];
		seq[V_front.order] = '\0';
		for (int jj=j; jj<V.order; jj++) {
			seq[jj-j] = 'a'+index[jj];
		}
		seq[V_front.order-1] = 'k';
		// make seq_w
		seq_w[0] = 'a'+index[j];
		// make seq_f
		char seq_f[V_front.order];
		seq_f[V_front.order-1] = '\0';
		for (int jj=j+1; jj<V.order; jj++) {
			seq_f[jj-j-1] = 'a'+index[jj];
		}
		seq_f[V_front.order-2] = 'k';
		// build len for V_temp
		int lens_V[V.order-j];
		for (int m=j; m<V.order-1; m++) {
			lens_V[m-j] = lens_H[m];
		}
		lens_V[V.order-j-1] = K;
		V_temp = Tensor<>(V.order-j, lens_V, dw);
		// contraction
		V_temp[seq_f] = V_front[seq]*W[index[j]][seq_w];
		V_front = V_temp;
	}
	M["ij"] = V_front["ij"];
	return;
}

/** 
 *  \brief subproblem grad_W[i]
 */
void gradsubprob(Matrix<>& M, 
				 Matrix<>& S, 
				 Matrix<>& W, 
				 Matrix<>& grad_W) {
	grad_W["ij"] = -M["ij"]+W["ik"]*S["kj"]; 
}

/**
 * \brief initialize grad_W
 */
void gradient_CP(Tensor<> & V, 
				 Matrix<> * W, 
				 Matrix<> * grad_W, 
				 World & dw) {
	//make the char
	char seq_V[V.order+1];
	seq_V[V.order] = '\0'; 
	for (int j=0; j<V.order; j++) {
		seq_V[j] = 'a'+j;
	}
	//initialize matrix S
	Matrix<> S = Matrix<>(W[0].ncol,W[0].ncol);
	// iteration on grad_W[i]
	for (int i=0; i<V.order; i++) { 
		//make the char
		char temp = seq_V[V.order-1];
		seq_V[V.order-1] = seq_V[i];
		seq_V[i] = temp;
		//construct H_T
		int lens_H[V.order];
		int index[V.order];
		for (int j=0; j<V.order-1; j++) {
			index[j] = (int)(seq_V[j]-'a');
			lens_H[j] = V.lens[index[j]];
		}
		index[V.order-1] = (int)(seq_V[V.order-1]-'a');
		lens_H[V.order-1] = W[i].ncol;
		//initialize matrix M
		Matrix<> M = Matrix<>(W[i].nrow,W[i].ncol);
		//Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
		KhatriRao_contract(M, V, W, index, lens_H, dw);
		//calculating S
		S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
		for (int ii=1; ii<V.order-1; ii++) {
			S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
		}
		//subproblem grad_W[i]
		gradsubprob(M, S, W[i], grad_W[i]);
		//recover the char
		temp = seq_V[V.order-1];
		seq_V[V.order-1] = seq_V[i];
		seq_V[i] = temp;
	}
}