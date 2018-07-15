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
	Tensor<> core_prev(core);
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

void Build_ttmc_map(map<string, Tensor<>> & ttmc_map, 
					Tensor<> & V, 
					Matrix<> * W,
					char* args,
					World & dw) {

	int level = strlen(args);
	Tensor<> M;
	if (level==1) M = V;
	else {
		char args2[sizeof(args)];
		strncpy(args2,args, strlen(args)-1);
		args2[strlen(args)-1] = '\0';
		if (ttmc_map.find(args2) == ttmc_map.end()) {
			Build_ttmc_map(ttmc_map, V, W, args2, dw);
		} 
		M = ttmc_map[args2];
	}
	char args3[3];
	char seq1_contract[V.order+1];
	char seq2_contract[V.order+1];
	seq1_contract[V.order] = '\0';
	seq2_contract[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq1_contract[jj] = 'a'+jj;
		seq2_contract[jj] = 'a'+jj;
	}
	seq1_contract[int(args[strlen(args)-1]-'a')] = '*';
	args3[0] = args[strlen(args)-1];
	args3[1] = '*';
	args3[2] = '\0';
	// initialize
	int lens[V.order];
	for (int ii=0; ii<strlen(seq1_contract); ii++){
		if (seq1_contract[ii] == '*') lens[ii] = W[ii].ncol;
		else lens[ii] = M.lens[ii];
	}
	ttmc_map[args] = Tensor<>(V.order, lens, dw);
	ttmc_map[args][seq1_contract] = M[seq2_contract] * W[int(args3[0]-'a')][args3];
}

/**
 * \brief ALS method for Tucker decomposition
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsTucker_mod(Tensor<> & V, 
				   Tensor<> & core, 
				   Matrix<> * W, 
				   double tol, 
				   double timelimit, 
				   int maxiter, 
				   World & dw) {

	double st_time = MPI_Wtime();
	int iter; 
	Tensor<> core_prev(core);
	double diffnorm;
	// initialize the char
	char seq[V.order+1], seq_Y[V.order+1], seq_dW[3];
	seq[V.order] = '\0';
	seq_Y[V.order] = '\0';
	seq_dW[2] = '\0';
	seq_dW[1] = '*';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_Y[jj] = 'a'+jj;
	}
	Matrix<> * dW = new Matrix<>[V.order];
	Matrix<> * W_init = new Matrix<>[V.order];
	// initialize the map
	map<string, Tensor<>> ttmc_map;
	for (iter=0; iter<=maxiter; iter++)
	{
		// initialize the TTMc
		if (iter%30==0) {
			for (int j=0; j<V.order; j++) {
				W_init[j] = W[j];
				dW[j] = Matrix<>(W[j].nrow,W[j].ncol);
				dW[j]["ij"] = 0.;
			}
			ttmc_map.clear();
			// build the char [abcd...] except ii and jj
			for (int ii=0; ii<V.order; ii++)
			for (int jj=ii+1; jj<V.order; jj++){
				char args[V.order-1];
				args[V.order-2] = '\0';
				strncpy(args,seq,ii);
				strncpy(args+ii,seq+ii+1,jj-ii-1);
				strncpy(args+jj-1,seq+jj+1,V.order-jj-1);
				Build_ttmc_map(ttmc_map, V, W, args, dw);
			}
			// build the char [abcd...] except ii
			for (int ii=0; ii<V.order; ii++) {
				char args[V.order];
				args[V.order-1] = '\0';
				strncpy(args,seq,ii);
				strncpy(args+ii,seq+ii+1,V.order-ii-1);
				//if(dw.rank==0) cout <<"args= "<< args << endl;
				Build_ttmc_map(ttmc_map, V, W, args, dw);
			}			
		}
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
			/* Compute the TTMc result Y
			*  Y["ijkd"] = V["abcd"]*R[0]["ai"]*R[1]["bj"]*R[2]["ck"]
			*/
			Tensor<> Y;
			//TTMc(Y, V, W, i, dw);
			char args_Y[V.order];
			args_Y[V.order-1] = '\0';
			strncpy(args_Y,seq,i);
			strncpy(args_Y+i,seq+i+1,V.order-i-1);
			Y = ttmc_map[args_Y];
			for (int ii=0;ii<i; ii++) {
				// build args
				char args[V.order-1];
				args[V.order-2] = '\0';
				strncpy(args,seq,ii);
				strncpy(args+ii,seq+ii+1,i-ii-1);
				strncpy(args+i-1,seq+i+1,V.order-i-1);
				// build seq
				seq_dW[0] = 'a'+ii;
				seq_Y[ii] = '*';
				Y[seq_Y] += ttmc_map[args][seq]*dW[ii][seq_dW];
				seq_Y[ii] = 'a'+ii;
			}
			for (int ii=i+1;ii<V.order; ii++) {
				// build args
				char args[V.order-1];
				args[V.order-2] = '\0';
				strncpy(args,seq,i);
				strncpy(args+i,seq+i+1,ii-i-1);
				strncpy(args+ii-1,seq+ii+1,V.order-ii-1);
				// build seq
				seq_dW[0] = 'a'+ii;
				seq_Y[ii] = '*';
				Y[seq_Y] += ttmc_map[args][seq]*dW[ii][seq_dW];	
				seq_Y[ii] = 'a'+ii;			
			}
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
			dW[i]["ij"] = W[i]["ij"]-W_init[i]["ij"];
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