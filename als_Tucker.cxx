/** \addtogroup examples 
  * @{ 
  * \defgroup als_tensor_factorization als_tensor_factorization
  * @{ 
  * \brief NTF algorithms based on projected gradient methods
  */

#include "common.h"
#include "als_Tucker.h"
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
	Tensor<> V_temp;
	V_temp = V;
	for (int index=0; index<V.order; index++) {
		if (index != i) {
			seq_p[0] = index+'a';
			seq[index] = 'k';
			//lens
			lens_Y[index] = W[index].ncol;
			Y = Tensor<>(V.order, lens_Y, dw);		
			Y[seq] = (V_temp)[seq_mod]*W[index][seq_p];
			// TODO: check memory leak
			V_temp = Tensor<>(V.order, lens_Y, dw);
			V_temp = Y;
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
			if(dw.rank==0) cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [diffnorm]  "<< diffnorm << "  [tol]  " << tol <<  "\n";
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

void ttmc_map_DT(map<string,Tensor<>>& ttmc_map, 
				 map<string,string>& parent, 
				 map<string,string>& sibling, 
				 Tensor<>& V, 
				 Matrix<> * W, 
				 string args,
				 World& dw) {

	if(ttmc_map.find(args)!=ttmc_map.end()) return;
	char seq_w[3];
	seq_w[2] = '\0'; seq_w[1] = '*'; 
	// initialize the char
	char seq[V.order+1];
	char seq_Y[V.order+1];
	seq[V.order] = '\0'; seq_Y[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_Y[jj] = 'a'+jj;
	}
	Tensor<> V_front;
	Tensor<> V_parent;
	if (args.length()==V.order/2 || args.length()==V.order/2+1) {
		V_front = V;
		V_parent = V;
	} else {
		if (ttmc_map.find(parent[args])==ttmc_map.end()) {
			ttmc_map_DT(ttmc_map, parent, sibling, V, W, parent[args], dw);
		}
		V_front = ttmc_map[parent[args]];
		V_parent = ttmc_map[parent[args]];
	}
	// build len for Y
	int lens_Y[V_parent.order];
	for (int m=0; m<V_parent.order; m++) {
		lens_Y[m] = V_parent.lens[m];
	}
	Tensor<> V_temp; 
	// int index_start = int(sibling[args][0]-parent[args][0]);
	/* loops */
	for (int j=0; j<sibling[args].length(); j++) {     // iterate on [ab]
		// make seq_w
		seq_w[0] = sibling[args][j];			
		// build len for V_temp
		lens_Y[int(seq_w[0]-'a')] = W[int(seq_w[0]-'a')].ncol;
		V_temp = Tensor<>(V_parent.order, lens_Y, dw);
		// contraction
		seq_Y[int(seq_w[0]-'a')] = '*';
		V_temp[seq_Y] = V_front[seq]*W[seq_w[0]-'a'][seq_w];	
		seq_Y[int(seq_w[0]-'a')] = seq_w[0];			
		V_front = V_temp;
	}
	ttmc_map[args] = V_front;
	return;
}

/**
 * \brief ALS method for Tucker decomposition with dimension tree
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsTucker_DT(Tensor<> & V, 
				  Tensor<> & core, 
				  Matrix<> * W, 
				  double tol, 
				  double timelimit, 
				  int maxiter, 
				  ofstream & Plot_File,
				  int resprint,
				  World & dw) {
	cout.precision(13);
    Plot_File << "[dim],[iter],[diffnorm],[tol],[pp_update],[diffV],[dtime]" << "\n";          //Headings for file

	double st_time = MPI_Wtime();
	int iter; 
	Tensor<> core_prev(core);
	double diffnorm = 1000;
	double diffnorm_V = 1000;
	// initialize the char
	char seq[V.order+1];
	char seq_Y[V.order+1];
	char seq_Y_end[V.order+1];
	seq[V.order] = '\0'; seq_Y[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_Y[jj] = 'a'+jj;
		seq_Y_end[jj] = 'a'+jj;
	}
	seq_Y_end[V.order-1] = '*';
	char seq_W_end[3];
	seq_W_end[0] = '*'; 
	seq_W_end[1] = seq[V.order-1]; 
	seq_W_end[2] = '\0'; 

	// maps 
	map<string, Tensor<>> ttmc_map;
	map<string, string> parent;
	map<string, string> sibling;
	Construct_Dimension_Tree(parent, sibling, 0, V.order-1);
	// build len for Y
	int lens_Y[V.order];
	for (int m=0; m<V.order; m++) {
		lens_Y[m] = W[m].ncol;
	}
	lens_Y[V.order-1] = V.lens[V.order-1];
	Tensor<> Y_end = Tensor<>(V.order, lens_Y, dw);	
	lens_Y[V.order-1] = W[V.order-1].ncol;
	// iterations
	for (iter=0; iter<=maxiter; iter++)
	{
		// print the difference norm 
		if ((iter%resprint==0 && iter!=0) || iter==maxiter) {
			double st_time1 = MPI_Wtime();

				TTMc(core, V, W, -1, dw);
				double diffnorm1 = core.norm2();
				double diffnorm2 = core_prev.norm2();
				diffnorm = abs(diffnorm1-diffnorm2);
				// check the residule
				Matrix<> W_T[V.order];
				for (int i=0; i<V.order; i++) {
					W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
					W_T[i]["ij"] = W[i]["ji"];
				}
				Tensor<> V_check(V.order, V.lens, dw);
				Tensor<> V_diff(V.order, V.lens, dw);
				TTMc(V_check, core, W_T, -1, dw);
				char seq[V.order+1];
				seq[V.order] = '\0';
				for (int jj=0; jj<V.order; jj++) {
					seq[jj] = 'a'+jj;
				}
				V_diff[seq] = V_check[seq] - V[seq];
				diffnorm_V = V_diff.norm2();

			st_time += MPI_Wtime() - st_time1;
			double dtime = MPI_Wtime() - st_time;
			if(dw.rank==0) {
				cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [diffnorm]  "<< diffnorm << "  [tol]  " << tol << "  [pp_update]  " << 0  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
				// plot to file
				Plot_File << V.lens[0] << "," << iter << "," << diffnorm << "," << tol << "," << 0 << "," << diffnorm_V << "," << dtime << "\n";
				if(iter%100==0 && iter!=0) {// flush
					Plot_File << endl;
				}
			}
			// end check the residue
			if ((diffnorm < tol) || MPI_Wtime()-st_time > timelimit) 
				break;
			core_prev[seq] = core[seq];
		}
		// clear the Hash Table
		ttmc_map.clear();
		// iteration on W[i]
		for (int i=0; i<V.order; i++) { 
			/* Compute the coarse level V 
			*  Y["ijkd"] = V["abcd"]*R[0]["ai"]*R[1]["bj"]*R[2]["ck"]
			*/
			// Tensor<> Y;
			// TTMc(Y, V, W, i, dw);
			// build len for Y
			// int lens_Y[V.order];
			// for (int m=0; m<V.order; m++) {
			// 	lens_Y[m] = W[m].ncol;
			// }
			lens_Y[i] = V.lens[i];
			Tensor<> Y = Tensor<>(V.order, lens_Y, dw);
			lens_Y[i] = W[i].ncol;	
			// make args
			char args[2];
			args[1] = '\0';
			args[0] = i+'a';
			if (ttmc_map.find(parent[args])==ttmc_map.end()) {
				ttmc_map_DT(ttmc_map, parent, sibling, V , W, parent[args], dw);
			}
			if (sibling[args].length()==1) {
				char seq_A[3];
				seq_A[2] = '\0'; seq_A[1] = '*'; 
				if (args[0]==parent[args][0]) seq_A[0] = parent[args][1];
				else seq_A[0] = parent[args][0];
				seq_Y[int(seq_A[0]-'a')] = '*';
				Y[seq_Y] = ttmc_map[parent[args]][seq]*W[int(seq_A[0]-'a')][seq_A];
				seq_Y[int(seq_A[0]-'a')] = seq_A[0];				
			} else {
				char seq_A1[3],seq_A2[3];
				seq_A1[2] = '\0'; seq_A2[2] = '\0';
				seq_A1[1] = '*'; seq_A2[1] = '^';
				if (parent[args][0]==args[0]) {
					seq_A1[0] = parent[args][1];
					seq_A2[0] = parent[args][2];
				}
				else {
					seq_A1[0] = parent[args][0];
					seq_A2[0] = parent[args][1];
				}
				seq_Y[int(seq_A1[0]-'a')] = '*';
				seq_Y[int(seq_A2[0]-'a')] = '^';
				Y[seq_Y] = ttmc_map[parent[args]][seq]*W[int(seq_A1[0]-'a')][seq_A1]*W[int(seq_A2[0]-'a')][seq_A2];	
				seq_Y[int(seq_A1[0]-'a')] = seq_A1[0];
				seq_Y[int(seq_A2[0]-'a')] = seq_A2[0];			
			}
			if (i==V.order-1) {
				Y_end[seq_Y] = Y[seq_Y];
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
		}
		core[seq] = Y_end[seq_Y_end] * W[V.order-1][seq_W_end];
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final Diff norm %E \n", iter, diffnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	Plot_File.close();
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
 * \brief ALS method for Tucker decomposition with dimension tree PP subroutine
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
void alsTucker_DT_sub(Tensor<> & V, 
					  Tensor<> & core, 
					  Tensor<> & core_prev,
					  Matrix<> * W, 
					  Matrix<> * dW,
					  double tol, 
					  double tol_init,
					  double timelimit, 
					  int maxiter, 
					  double & st_time,
					  ofstream & Plot_File,
					  double & diffnorm,
					  int & iter,
					  int resprint,
					  World & dw) {

	// work as the preconditioning of pairwise perturbation
	Matrix<> W_prev[V.order];
	for (int i=0; i<V.order; i++) {
		W_prev[i] = Matrix<>(W[i].nrow,W[i].ncol);
	}
	// initialize the char
	char seq[V.order+1];
	char seq_Y[V.order+1];
	char seq_Y_end[V.order+1];
	double diffnorm_V = 1000;
	seq[V.order] = '\0'; seq_Y[V.order] = '\0';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_Y[jj] = 'a'+jj;
		seq_Y_end[jj] = 'a'+jj;
	}
	seq_Y_end[V.order-1] = '*';
	char seq_W_end[3];
	seq_W_end[0] = '*'; 
	seq_W_end[1] = seq[V.order-1]; 
	seq_W_end[2] = '\0'; 
	// maps 
	map<string, Tensor<>> ttmc_map;
	map<string, string> parent;
	map<string, string> sibling;
	Construct_Dimension_Tree(parent, sibling, 0, V.order-1);
	// build len for Y
	int lens_Y[V.order];
	for (int m=0; m<V.order; m++) {
		lens_Y[m] = W[m].ncol;
	}
	lens_Y[V.order-1] = V.lens[V.order-1];
	Tensor<> Y_end = Tensor<>(V.order, lens_Y, dw);	
	lens_Y[V.order-1] = W[V.order-1].ncol;

	// iterations
	for (; iter<=maxiter; iter++)
	{
		// print the difference norm 
		if ((iter%resprint==0 && iter!=0) || iter==maxiter ) {
			double st_time1 = MPI_Wtime();

				TTMc(core, V, W, -1, dw);
				double diffnorm1 = core.norm2();
				double diffnorm2 = core_prev.norm2();
				diffnorm = abs(diffnorm1-diffnorm2);
				// check the residule
				Matrix<> W_T[V.order];
				for (int i=0; i<V.order; i++) {
					W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
					W_T[i]["ij"] = W[i]["ji"];
				}
				Tensor<> V_check(V.order, V.lens, dw);
				Tensor<> V_diff(V.order, V.lens, dw);
				TTMc(V_check, core, W_T, -1, dw);
				char seq[V.order+1];
				seq[V.order] = '\0';
				for (int jj=0; jj<V.order; jj++) {
					seq[jj] = 'a'+jj;
				}
				V_diff[seq] = V_check[seq] - V[seq];
				diffnorm_V = V_diff.norm2();
			
			st_time += (MPI_Wtime() - st_time1);
			double dtime = MPI_Wtime() - st_time;
			if(dw.rank==0) { 
				cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [diffnorm]  "<< diffnorm << "  [tol]  " << tol << "  [pp_update]  " << "0"  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
				// plot to file
				Plot_File << V.lens[0] << "," << iter << "," << diffnorm << "," << tol << "," << 0 << "," << diffnorm_V << "," << dtime << "\n";
				if(iter%100==0 && iter!=0) {// flush
					Plot_File << endl;
				}
			}			
			// end check the residue
			if ((diffnorm < tol) || MPI_Wtime()-st_time > timelimit) 
				break;
			core_prev[seq] = core[seq];
		}
		// clear the Hash Table
		ttmc_map.clear();
		// iteration on W[i]
		for (int i=0; i<V.order; i++) { 
			/* Compute the coarse level V 
			*  Y["ijkd"] = V["abcd"]*R[0]["ai"]*R[1]["bj"]*R[2]["ck"]
			*/
			// Tensor<> Y;
			// TTMc(Y, V, W, i, dw);
			// build len for Y
			int lens_Y[V.order];
			for (int m=0; m<V.order; m++) {
				lens_Y[m] = W[m].ncol;
			}
			lens_Y[i] = V.lens[i];
			Tensor<> Y = Tensor<>(V.order, lens_Y, dw);	
			// make args
			char args[2];
			args[1] = '\0';
			args[0] = i+'a';
			if (ttmc_map.find(parent[args])==ttmc_map.end()) {
				ttmc_map_DT(ttmc_map, parent, sibling, V , W, parent[args], dw);
			}
			if (sibling[args].length()==1) {
				char seq_A[3];
				seq_A[2] = '\0'; seq_A[1] = '*'; 
				if (args[0]==parent[args][0]) seq_A[0] = parent[args][1];
				else seq_A[0] = parent[args][0];
				seq_Y[int(seq_A[0]-'a')] = '*';
				Y[seq_Y] = ttmc_map[parent[args]][seq]*W[int(seq_A[0]-'a')][seq_A];
				seq_Y[int(seq_A[0]-'a')] = seq_A[0];				
			} else {
				char seq_A1[3],seq_A2[3];
				seq_A1[2] = '\0'; seq_A2[2] = '\0';
				seq_A1[1] = '*'; seq_A2[1] = '^';
				if (parent[args][0]==args[0]) {
					seq_A1[0] = parent[args][1];
					seq_A2[0] = parent[args][2];
				}
				else {
					seq_A1[0] = parent[args][0];
					seq_A2[0] = parent[args][1];
				}
				seq_Y[int(seq_A1[0]-'a')] = '*';
				seq_Y[int(seq_A2[0]-'a')] = '^';
				Y[seq_Y] = ttmc_map[parent[args]][seq]*W[int(seq_A1[0]-'a')][seq_A1]*W[int(seq_A2[0]-'a')][seq_A2];	
				seq_Y[int(seq_A1[0]-'a')] = seq_A1[0];
				seq_Y[int(seq_A2[0]-'a')] = seq_A2[0];			
			}
			if (i==V.order-1) {
				Y_end[seq_Y] = Y[seq_Y];
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
			Matrix<> check = Matrix<>(W[i].ncol,W[i].ncol);
			Matrix<> trans = Matrix<>(W[i].ncol,W[i].ncol);
			check["ik"] = W[i]["ji"]*W_prev[i]["jk"];
			S["i"] = check["ii"];
			Transform<double,double>([](double & b){ if (b>0) b=1; else b=-1; })(S["i"]);
			trans["ii"] = S["i"];
			W[i]["ij"] = W[i]["ik"]*trans["kj"];

		}
		core[seq] = Y_end[seq_Y_end] * W[V.order-1][seq_W_end];

		// work as the preconditioning of pairwise perturbation
		int num_dw_break = 0;
		for (int i=0; i<V.order; i++) {
			dW[i]["ij"] = W[i]["ij"] - W_prev[i]["ij"];
			W_prev[i]["ij"] = W[i]["ij"];
			double norm_dW = dW[i].norm2();
			// if (dw.rank==0) cout << norm_dW << endl;
			double norm_W = W[i].norm2();
			if (abs(norm_dW/norm_W)<tol_init) num_dw_break++;
		}
		if (num_dw_break==V.order) return;
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	// if(dw.rank==0) {
	// 	printf ("\nIter = %d Final Diff norm %E \n", iter, diffnorm);
	// 	printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	// }
	return;
}

/**
 * \brief ALS method for Tucker decomposition
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
void alsTucker_PP_sub(Tensor<> & V, 
				  Tensor<> & core, 
				  Tensor<> & core_prev,
				  Matrix<> * W, 
				  Matrix<> * dW,
				  double tol, 
				  double tol_init,
				  double timelimit, 
				  int maxiter, 
				  double & st_time,
				  ofstream & Plot_File,
				  double & diffnorm,
				  int & iter,
				  int resprint,
				  World & dw) {

	int init_iter = iter;
	double diffnorm_V = 1000;
	// initialize the char
	char seq[V.order+1], seq_Y[V.order+1], seq_dW[3];
	char seq_Y_end[V.order+1];
	seq[V.order] = '\0';
	seq_Y[V.order] = '\0';
	seq_dW[2] = '\0';
	seq_dW[1] = '*';
	for (int jj=0; jj<V.order; jj++) {
		seq[jj] = 'a'+jj;
		seq_Y[jj] = 'a'+jj;
		seq_Y_end[jj] = 'a'+jj;
	}
	seq_Y_end[V.order-1] = '*';
	char seq_W_end[3];
	seq_W_end[0] = '*'; 
	seq_W_end[1] = seq[V.order-1]; 
	seq_W_end[2] = '\0'; 

	Matrix<> W_init[V.order];
	// initialize the map
	map<string, Tensor<>> ttmc_map;

	// build len for Y
	int lens_Y[V.order];
	for (int m=0; m<V.order; m++) {
		lens_Y[m] = W[m].ncol;
	}
	lens_Y[V.order-1] = V.lens[V.order-1];
	Tensor<> Y_end = Tensor<>(V.order, lens_Y, dw);	
	lens_Y[V.order-1] = W[V.order-1].ncol;

	for (; iter<=maxiter; iter++)
	{
		// work as the preconditioning of pairwise perturbation
		int num_dw_break = 0;
		for (int i=0; i<V.order; i++) {
			double norm_dW = dW[i].norm2();
			double norm_W = W[i].norm2();
			if (abs(norm_dW/norm_W)>tol_init) num_dw_break++;
		}
		// initialize the TTMc
		// if (iter==0 || iter == (iter_prev+pp_update)) {
		if (iter==init_iter || (num_dw_break > 0 ) ) {
			if (num_dw_break>0) {
				return;
			}
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
				Build_ttmc_map(ttmc_map, V, W, args, dw);
			}	
		}
		// print the difference norm 
		if ((iter%resprint==0 && iter!=0) || iter==maxiter || iter==init_iter) {
			double st_time1 = MPI_Wtime();

				TTMc(core, V, W, -1, dw);
				double corenorm = core.norm2();
				double corenorm_prev = core_prev.norm2();
				diffnorm = abs(corenorm-corenorm_prev);
				// check the residule
				Matrix<> W_T[V.order];
				for (int i=0; i<V.order; i++) {
					W_T[i] = Matrix<>(W[i].ncol,W[i].nrow,dw);
					W_T[i]["ij"] = W[i]["ji"];
				}
				Tensor<> V_check(V.order, V.lens, dw);
				Tensor<> V_diff(V.order, V.lens, dw);
				TTMc(V_check, core, W_T, -1, dw);
				char seq[V.order+1];
				seq[V.order] = '\0';
				for (int jj=0; jj<V.order; jj++) {
					seq[jj] = 'a'+jj;
				}
				V_diff[seq] = V_check[seq] - V[seq];
				diffnorm_V = V_diff.norm2();

			st_time += (MPI_Wtime() - st_time1);
			double dtime = MPI_Wtime() - st_time;
			if(dw.rank==0) {
				cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [diffnorm]  "<< diffnorm << "  [tol]  " << tol << "  [pp_update]  " << 1  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
				// plot to file
				Plot_File << V.lens[0] << "," << iter << "," << diffnorm << "," << tol << "," << 1 << "," << diffnorm_V << "," << dtime << "\n";
				if(iter%100==0 && iter!=0) {// flush
					Plot_File << endl;
				}
			}
			// end check the residue
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
			if (i==V.order-1) {
				Y_end[seq_Y] = Y[seq_Y];
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

			Matrix<> check = Matrix<>(W[i].ncol,W[i].ncol);
			Matrix<> trans = Matrix<>(W[i].ncol,W[i].ncol);
			check["ik"] = W[i]["ji"]*W_init[i]["jk"];
			S["i"] = check["ii"];
			Transform<double,double>([](double & b){ if (b>0) b=1; else b=-1; })(S["i"]);
			trans["ii"] = S["i"];
			W[i]["ij"] = W[i]["ik"]*trans["kj"];

			dW[i]["ij"] = W[i]["ij"]-W_init[i]["ij"];

			double norm_dW = dW[i].norm2();
		}
		core[seq] = Y_end[seq_Y_end] * W[V.order-1][seq_W_end];
		// print .
		// if (iter%10==0 && dw.rank==0) printf(".");
	}
	// if(dw.rank==0) {
	// 	printf ("\nIter = %d Final Diff norm %E \n", iter, diffnorm);
	// 	printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	// }
	// Plot_File.close();
	return;
}

/**
 * \brief ALS method for Tucker decomposition
 *  W: output matrices
 *  core: output core tensor
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsTucker_PP(Tensor<> & V, 
				  Tensor<> & core, 
				  Matrix<> * W, 
				  double tol, 
				  double tol_init,
				  double timelimit, 
				  int maxiter, 
				  ofstream & Plot_File,
				  int resprint,
				  World & dw) {
	cout.precision(13);
    Plot_File << "[dim],[iter],[diffnorm],[tol],[pp_update],[diffV],[dtime]" << "\n";          //Headings for file

	double st_time = MPI_Wtime();
	int iter = 0;
	// if (dw.rank==0) printf("pairwise perturbation starts from %d\n", iter);
	// if (dw.rank==0) printf("pairwise perturbation restarts from %d\n", iter);
	Tensor<> core_prev(core);
	double diffnorm = 10.;
	// initialize dW
	Matrix<> * dW = new Matrix<>[V.order];
	for (int j=0; j<V.order; j++) {
		dW[j] = Matrix<>(W[j].nrow,W[j].ncol);
		dW[j]["ij"] = 0.;
	}

	while (diffnorm > tol && iter<=maxiter)
	{
		
		if (dw.rank==0) printf("DT starts from %d\n", iter);

		alsTucker_DT_sub(V, core, core_prev,
					  	 W, dW,
					  	 tol, tol_init,	
					  	 timelimit, maxiter, 
					  	 st_time, Plot_File,
					  	 diffnorm, iter, resprint, dw);

		if (dw.rank==0) printf("pairwise perturbation starts from %d\n", iter);

		alsTucker_PP_sub(V, core, core_prev,
						 W, dW,
						 tol, tol_init,
						 timelimit, maxiter,
						 st_time, Plot_File,
						 diffnorm, iter, resprint, dw);

		if (tol_init>5e-3) tol_init *= 0.9;

	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final Diff norm %E \n", iter, diffnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	Plot_File.close();
	delete[] dW;
	if (iter == maxiter+1) return false;
	else return true;
}
