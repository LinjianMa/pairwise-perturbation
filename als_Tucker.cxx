/** \addtogroup examples 
  * @{ 
  * \defgroup als_tensor_factorization als_tensor_factorization
  * @{ 
  * \brief NTF algorithms based on projected gradient methods
  */
#include <ctf.hpp>
#include "als_CP.cxx"
using namespace CTF;
//#define ERR_REPORT

void get_factor_matrices(Tensor<>& T, 
						 Matrix<>* factor_matrices,
						 int ranks[], 
						 World& dw) {
  
	char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
	char arg[T.order+1];
	int transformed_lens[T.order];
	char transformed_arg[T.order+1];
	transformed_arg[T.order] = '\0';
	for (int i = 0; i < T.order; i++) {
		arg[i] = chars[i];
		transformed_arg[i] = chars[i];
		transformed_lens[i] = T.lens[i];
	}
	arg[T.order] = '\0';

	for (int i = 0; i < T.order; i++) {
		for (int j = i; j > 0; j--) { 
			transformed_lens[j] = T.lens[j-1];
		}

		transformed_lens[0] = T.lens[i];
		for (int j = 0; j < i; j++) {
			transformed_arg[j] = arg[j+1];
		}
		transformed_arg[i] = arg[0];

		int unfold_lens [2];
		unfold_lens[0] = T.lens[i];
		int ncol = 1;

		for (int j = 0; j < T.order; j++) {  
			if (j != i) ncol *= T.lens[j];  
    	}
    	unfold_lens[1] = ncol;

		Tensor<double> transformed_T(T.order, transformed_lens, dw);
		transformed_T[arg] = T[transformed_arg];

		Tensor<double> cur_unfold(2, unfold_lens, dw);
		fold_unfold(transformed_T, cur_unfold);

		Matrix<double> M(cur_unfold);
		Matrix<> U;
		Matrix<> VT;
		Vector<> S;
		M.svd(U, S, VT, ranks[i]);

	    factor_matrices[i] = U;
  }
}

Tensor<> get_core_tensor(Tensor<>& T, 
					 	 Matrix<>* factor_matrices, 
					 	 int ranks[], 
					 	 World& dw) {

	std::vector<Tensor<>> core_tensors(T.order+1);
	core_tensors[0] = T;
	int lens[T.order];
	for (int i = 0; i < T.order; i++) {
		lens[i] = T.lens[i];
	} 
	for (int i = 1; i < T.order+1; i++) {
		lens[i-1] = ranks[i-1];
		Tensor<double> core(T.order, lens, dw);
		core_tensors[i] = core;   
	}
	//calculate core tensor
	char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
	char arg[T.order+1];
	char core_arg[T.order+1];
	for (int i = 0; i < T.order; i++) {
 		arg[i] = chars[i];
		core_arg[i] = chars[i];
	}
 	arg[T.order] = '\0';
	core_arg[T.order] = '\0';
	char matrix_arg[3];
	matrix_arg[0] = 'a';
	matrix_arg[2] = '\0';
	for (int i = 0; i < T.order; i++) {
		core_arg[i] = 'a';
		matrix_arg[1] = arg[i];
		Matrix<double> transpose(factor_matrices[i].ncol, factor_matrices[i].nrow, dw);
		transpose["ij"] = factor_matrices[i]["ji"];
		core_tensors[i+1][core_arg] = transpose[matrix_arg] * core_tensors[i][arg];
		core_arg[i] = arg[i];
	}
	return core_tensors[T.order];
}

void hosvd(Tensor<>& T, 
	 	   Tensor<>& core, 
	 	   Matrix<>* factor_matrices, 
	 	   int * ranks, 
	 	   World& dw) {
	get_factor_matrices(T, factor_matrices, ranks, dw);
	core = Tensor<double>(get_core_tensor(T, factor_matrices, ranks, dw)); 
}

/* Doing Tensor Times Matrix contraction
*  except index i
*  i=-1 : contract all the indices
*/
void TTMc(Tensor<>& Y, 
		  Tensor<>& V,
		  Matrix<>* W, 
		  int i, 
		  World & dw){

	char seq[V.order+1], seq_mod[V.order+1], seq_p[3];
	seq[V.order] = '\0';
	seq_mod[V.order] = '\0';
	seq_p[2] = '\0';
	seq_p[1] = 'k';
	// make seq
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_mod[jj] = 'a'+jj;
	}
	// build len for Y
	int lens_Y[V.order];
	for (int m=0; m<V.order; m++) {
		lens_Y[m] = V.lens[m];
	}
	Tensor<> * V_temp = new Tensor<>;
	*V_temp = V;
	for (int index=0; index<V.order; index++) {
		if (index != i) {
			seq_p[0] = index+'a';
			seq[index] = 'k';
			//lens
			lens_Y[index] = W[index].ncol;
			Y = Tensor<>(V.order, lens_Y, dw);		
			Y[seq] = (*V_temp)[seq_mod]*W[index][seq_p];
			V_temp = new Tensor<>(V.order, lens_Y, dw);
			*V_temp = Y;
			//recover seq
			seq[index] = 'a'+index;
		}
	}
}

/**
 * \brief ALS method for Tucker decomposition
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsTucker(Tensor<> & V, 
			   Tensor<> & core, 
			   Matrix<> * W, 
			   double tol, 
			   double timelimit, 
			   int maxiter, 
			   World & dw) {

	double st_time = MPI_Wtime();
	int iter; 
	Tensor<> core_prev(core), core_diff(core);
	double diffnorm;
	// initialize the char
	char seq[V.order+1];
	seq[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
	}
	for (iter=0; iter<=maxiter; iter++)
	{
		// print the difference norm 
		if ((iter%100==0 && iter!=0) || iter==maxiter) {
			TTMc(core, V, W, -1, dw);
			double diffnorm1 = core.norm2();
			double diffnorm2 = core_prev.norm2();
			diffnorm = abs(diffnorm1-diffnorm2);
			if(dw.rank==0) cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [diffnorm]  "<< diffnorm << "  [tol]  " << tol <<  endl;
			if ((diffnorm < tol) || MPI_Wtime()-st_time > timelimit) 
				break;
			core_prev[seq] = core[seq];
		}
		// iteration on W[i]
		for (int i=0; i<V.order; i++) { 
			/* Compute the coarse level V 
			*  Y["ijkd"] = V["abcd"]*R[0]["ai"]*R[1]["bj"]*R[2]["ck"]
			*/
			Tensor<> Y;
			TTMc(Y, V, W, i, dw);
			/* transpose Y
			*/
			// seq setup
			char transformed_seq[Y.order+1];
			strncpy(transformed_seq,seq, strlen(seq)+1);
			transformed_seq[0] = seq[i];
			strncpy(transformed_seq+1,seq, i);
			// lens setup
			int lens_transform_Y[Y.order];
			for (int ii = 0; ii <Y.order; ii++) {
				lens_transform_Y[ii] = Y.lens[ii];
			}  
			for (int jj = i; jj > 0; jj--) { 
				lens_transform_Y[jj] = Y.lens[jj-1];
			}
			lens_transform_Y[0] = Y.lens[i];
			// build transformed_Y
			Tensor<> transformed_Y(Y.order, lens_transform_Y, dw);
			transformed_Y[transformed_seq] = Y[seq];
			/* unfold transformed_Y
			*/
			int unfold_lens[2];
			unfold_lens[0] = Y.lens[i];
			int unfold_ncol = 1;
			for (int jj = 0; jj < Y.order; jj++) {  
				if (jj != i) unfold_ncol *= Y.lens[jj];  
			}
			unfold_lens[1] = unfold_ncol;
			Tensor<> Y_unfold(2, unfold_lens, dw);
			fold_unfold(transformed_Y, Y_unfold);
			/* get leading singular vectors
			*/
			Matrix<> M(Y_unfold);
			Matrix<> U, VT;
			Vector<> S;
			Matrix<> MTM(M.nrow,M.nrow);
			MTM["ij"] = M["ik"]*M["jk"];
			MTM.svd(U, S, VT, core.lens[i]);
			double norm_U = U.norm2();
			if (norm_U<0) U["ij"] = -U["ij"];
			W[i] = U;
		}
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final Diff norm %E \n", iter, diffnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	if (iter == maxiter+1) return false;
	else return true;
}

// [cd] --> [ab*]
void stringbuilder_ttmc(char* seq, 
				   char* seq_return,
				   int N, 
				   World & dw) {

	char seq_complete[N+1];
	for (int ii=0; ii<N; ii++) {
		seq_complete[ii] = 'a'+ii; 
	}
	seq_complete[N] = '\0';
	if(seq=="0") {
		strncpy(seq_return,seq_complete, strlen(seq_complete)+1);
		return;
	} 
	char seq_contract[N+2-strlen(seq)];
	seq_contract[N+1-strlen(seq)] = '\0';
	seq_contract[N-strlen(seq)] = '*';	
	int jj = 0;
	int kk = 0;
	// build seq_return
	for (int ii=0; ii<N; ii++) {
		if (jj<strlen(seq) && seq_complete[ii]==seq[jj]){
			jj++;
			continue;
		}
		seq_contract[kk] = seq_complete[ii];
		kk++;
	}
	strncpy(seq_return,seq_contract, strlen(seq_contract)+1);
	return;
}

void Build_ttmc_map(map<string, Tensor<>> & mttkrp_map, 
					  Tensor<> & V, 
					  Matrix<> * W,
					  char* seq,
					  World & dw) {

	int level = strlen(seq);
	char seq3[3];
	// level=1 means it's the first contraction
	// For example: when seq = "a"
	// M[bcd*] = V[abcd]*W[a*]
	if (level==1) {
		char seq1_contract[V.order+1];
		stringbuilder_mttkrp(seq, seq1_contract, V.order, dw);
		char seq2_contract[V.order+1];
		stringbuilder_mttkrp("0", seq2_contract, V.order, dw);	
		char seq3_contract[3];
		seq3[0] = seq[0];
		seq3[1] = '*';
		seq3[2] = '\0';
		// initialize M[bcd*]
		int lens[strlen(seq1_contract)];
		for (int ii=0; ii<strlen(seq1_contract); ii++){
			if (seq1_contract[ii] == '*') lens[ii] = W[0].ncol;
			else lens[ii] = V.lens[int(seq1_contract[ii]-'a')];
		}
		mttkrp_map[seq] = Tensor<>(strlen(seq1_contract), lens, dw);
		mttkrp_map[seq][seq1_contract] = V[seq2_contract] * W[int(seq3[0]-'a')][seq3];
		return;
	}
	// level!=1 means it's the Khatri Rao product
	// For example: when seq = "bd"
	// M[ac*] = V[acd*]*W[d*]
	char seq2[sizeof(seq)];
	strncpy(seq2,seq, strlen(seq)-1);
	seq2[strlen(seq)-1] = '\0';
	if (mttkrp_map.find(seq2) == mttkrp_map.end()) {
		Build_mttkrp_map(mttkrp_map, V, W, seq2, dw);
	} 
	char seq1_contract[V.order+1];
	stringbuilder_mttkrp(seq, seq1_contract, V.order, dw);
	char seq2_contract[V.order+1];
	stringbuilder_mttkrp(seq2, seq2_contract, V.order, dw);	
	seq3[0] = seq[strlen(seq)-1];
	seq3[1] = '*';
	seq3[2] = '\0';
	// initialize M[ac*]
	int lens[strlen(seq1_contract)];
	for (int ii=0; ii<strlen(seq1_contract); ii++){
		if (seq1_contract[ii] == '*') lens[ii] = W[0].ncol;
		else lens[ii] = V.lens[int(seq1_contract[ii]-'a')];
	}
	mttkrp_map[seq] = Tensor<>(strlen(seq1_contract), lens, dw);
	mttkrp_map[seq][seq1_contract] = mttkrp_map[seq2][seq2_contract] * W[int(seq3[0]-'a')][seq3];

}

/**
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *	F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsTucker_mod(Tensor<> & V, 
			   Matrix<> * W, 
			   Matrix<> * grad_W, 
			   Matrix<> * F,
			   double tol, 
			   double timelimit, 
			   int maxiter, 
			   World & dw) {

	double st_time = MPI_Wtime();
	int iter; double projnorm; double Fnorm; 
	Matrix<> * grad_W_proj = new Matrix<>[V.order];
	// initialize the dW matrices
	Matrix<> * dW = new Matrix<>[V.order];
	Matrix<> * W_init = new Matrix<>[V.order];
	for (int j=0; j<V.order; j++) {
		dW[j] = Matrix<>(W[j].nrow, W[j].ncol);
	}
	// initialize the map
	map<string, Tensor<>> mttkrp_map;
	//make the char
	char seq[V.order+1], seq_V[V.order+1];
	seq[V.order] = '\0'; seq_V[V.order] = '\0'; 
	for (int j=0; j<V.order; j++) {
		seq[j] = 'a'+j;
		seq_V[j] = seq[j];
	}
	/*  initialize matrix S
	*	S["ij"] = W[0]["ki"]*W[0]["kj"]*W[1]["ki"]*W[1]["kj"]*W[2]["ki"]*W[2]["kj"]*W[3]["ki"]*...
	*/
	Matrix<> S = Matrix<>(W[0].ncol,W[0].ncol);

	for (iter=0; iter<=maxiter; iter++)
	{ 
		// initialize the MTTKRP
		if (iter%50==0) {
			for (int j=0; j<V.order; j++) {
				W_init[j] = W[j];
				dW[j]["ij"] = 0.;
			}
			mttkrp_map.clear();
			// build the char [abcd...] except ii and jj
			for (int ii=0; ii<V.order; ii++)
			for (int jj=ii+1; jj<V.order; jj++){
				char seq_tensor[V.order-1];
				seq_tensor[V.order-2] = '\0';
				strncpy(seq_tensor,seq,ii);
				strncpy(seq_tensor+ii,seq+ii+1,jj-ii-1);
				strncpy(seq_tensor+jj-1,seq+jj+1,V.order-jj-1);
				Build_mttkrp_map(mttkrp_map, V, W, seq_tensor, dw);
			}
			// build the char [abcd...] except ii
			for (int ii=0; ii<V.order; ii++) {
				char seq_tensor[V.order];
				seq_tensor[V.order-1] = '\0';
				strncpy(seq_tensor,seq,ii);
				strncpy(seq_tensor+ii,seq+ii+1,V.order-ii-1);
				//if(dw.rank==0) cout <<"seq_tensor= "<< seq_tensor << endl;
				Build_mttkrp_map(mttkrp_map, V, W, seq_tensor, dw);
			}			
		}
		// print the gradient norm 
		if (iter%100==0 || iter==maxiter) {
			//get the gradient
			gradient_CP(V, W, grad_W, dw);
			for (int i=0; i<V.order; i++) { 
				grad_W_proj[i] = Matrix<>(W[i].nrow,W[i].ncol);
				grad_W_proj[i]["ij"] = grad_W[i]["ij"]-F[i]["ij"];
			}
			projnorm = 0; Fnorm = 0;
			for (int i=0; i<V.order; i++) { 
				projnorm += grad_W_proj[i].norm2()*grad_W_proj[i].norm2();
				Fnorm += F[i].norm2();
			}
			projnorm = sqrt(projnorm);
			if(dw.rank==0) cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [projnorm]  "<< projnorm << "  [tol]  " << tol << "  [Fnorm]  " << Fnorm <<  endl;
			if ((projnorm < tol) || MPI_Wtime()-st_time > timelimit) 
				break;
		}
		// iteration on W[i]
		for (int i=0; i<V.order; i++) { 
			//make the char
			char temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;
			/*  construct Matrix M
			*	M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
			*/
			int lens_H[V.order];
			int index[V.order];
			for (int j=0; j<V.order-1; j++) {
				index[j] = (int)(seq_V[j]-'a');
				lens_H[j] = V.lens[index[j]];
			}
			index[V.order-1] = (int)(seq_V[V.order-1]-'a');
			lens_H[V.order-1] = W[i].ncol;
			// initialize matrix M
			Matrix<> M = Matrix<>(W[i].nrow,W[i].ncol);
			// Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
			//KhatriRao_contract(M2, V, W, index, lens_H, dw);
			char seq_M[V.order];
			strncpy(seq_M,seq,i);
			strncpy(seq_M+i,seq+i+1,V.order-i-1);
			//if(dw.rank==0) cout <<"seq_M= "<< seq_M << endl;
			M["ij"] = mttkrp_map[seq_M]["ij"];
			for (int ii=0;ii<i; ii++) {
				char seq_tensor[V.order-1];
				seq_tensor[V.order-2] = '\0';
				strncpy(seq_tensor,seq,ii);
				strncpy(seq_tensor+ii,seq+ii+1,i-ii-1);
				strncpy(seq_tensor+i-1,seq+i+1,V.order-i-1);
				M["jk"] += mttkrp_map[seq_tensor]["ijk"]*dW[ii]["ik"];
			}
			for (int ii=i+1;ii<V.order; ii++) {
				char seq_tensor[V.order-1];
				seq_tensor[V.order-2] = '\0';
				strncpy(seq_tensor,seq,i);
				strncpy(seq_tensor+i,seq+i+1,ii-i-1);
				strncpy(seq_tensor+ii-1,seq+ii+1,V.order-ii-1);
				M["ik"] += mttkrp_map[seq_tensor]["ijk"]*dW[ii]["jk"];				
			}			
			// calculating S
			S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
			for (int ii=1; ii<V.order-1; ii++) {
				S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
			}
			// subproblem M=W*S
			M["ij"] += F[i]["ij"];
			SVD_solve_mod(M, W[i], W_init[i], dW[i], S);
			// recover the char
			temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;
		}
		if (Fnorm == 0) Normalize(W, V.order, dw);
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	//if(dw.rank==0) {
		//printf ("\nIter = %d Final proj-grad norm %E \n", iter, projnorm);
		//printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	//}
	if (iter == maxiter+1) return false;
	else return true;
}