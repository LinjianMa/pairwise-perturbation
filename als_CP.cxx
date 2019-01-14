/** \addtogroup examples
  * @{
  * \defgroup als_tensor_factorization als_tensor_factorization
  * @{
  * \brief NTF algorithms based on projected gradient methods
  */
#include "common.h"
#include "als_CP.h"
//#define ERR_REPORT

/**
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *	F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP(Tensor<> & V,
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
			if(dw.rank==0) cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [projnorm]  "<< projnorm << "  [tol]  " << tol << "  [Fnorm]  " << Fnorm <<  "\n";
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
			KhatriRao_contract(M, V, W, index, lens_H, dw);
			// calculating S
			S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
			for (int ii=1; ii<V.order-1; ii++) {
				S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
			}
			// subproblem M=W*S
			M["ij"] += F[i]["ij"];
			SVD_solve(M, W[i], S);
			// Gauss_Seidel(W[i], M, S, 20);
			// recover the char
			temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;
		}
		if (Fnorm == 0) Normalize(W, V.order, dw);
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final proj-grad norm %E \n", iter, projnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	delete[] grad_W_proj;
	if (iter == maxiter+1) return false;
	else return true;
}

void mttkrp_map_DT(map<string,Tensor<>>& mttkrp_map,
				   map<string,string>& parent,
				   map<string,string>& sibling,
				   Tensor<>& V,
				   Matrix<> * W,
				   string args,
				   World& dw) {
	int K = W[0].ncol;
	if(mttkrp_map.find(args)!=mttkrp_map.end()) return;
	char seq_w[3];
	seq_w[2] = '\0'; seq_w[1] = '*';
	if(args.length()==V.order/2 || args.length()==V.order/2+1) {
		Tensor<> V_front = V;
		Tensor<> V_temp;
		/* initial condition
		*/
		char seq[V_front.order+1], seq_f[V_front.order+1];
		seq[V_front.order] = '\0';
		char_string_copy(seq,0,parent[args],0,parent[args].length());
		seq_f[V_front.order] = '\0';
		seq_w[0] = sibling[args][0];
		// make seq_f
		int index_start = int(sibling[args][0]-'a');
		char_string_copy(seq_f,0,parent[args],0,index_start);
		char_string_copy(seq_f,index_start,parent[args],index_start+1,V.order-index_start-1);
		seq_f[V.order-1] = '*';
		// build len for V_temp
		int lens_V[V.order];
		for (int m=0; m<index_start; m++) {
			lens_V[m] = V.lens[m];
		}
		for (int m=index_start+1; m<V.order; m++) {
			lens_V[m-1] = V.lens[m];
		}
		lens_V[V.order-1] = K;
		V_temp = Tensor<>(V.order, lens_V, dw);
		// contraction
		V_temp[seq_f] = V_front[seq]*W[index_start][seq_w];
		V_front = V_temp;
		/* loops
		*/
		for (int j=1; j<sibling[args].length(); j++) {     // iterate on [ab]
			// make seq
			seq[V_front.order] = '\0';
			strncpy(seq,seq_f, strlen(seq_f));
			// make seq_w
			seq_w[0] = sibling[args][j];
			// make seq_f
			seq_f[V_front.order-1] = '\0';
			seq_f[V_front.order-2] = '*';
			char_string_copy(seq_f,0,parent[args],0,index_start);
			char_string_copy(seq_f,index_start,parent[args],index_start+j+1,V.order-index_start-j-1);
			// build len for V_temp
			int lens_V[V.order-j];
			for (int m=0; m<index_start; m++) {
				lens_V[m] = V.lens[m];
			}
			for (int m=index_start+j+1; m<V.order; m++) {
				lens_V[m-j-1] = V.lens[m];
			}
			lens_V[V.order-j-1] = K;
			V_temp = Tensor<>(V.order-j, lens_V, dw);
			// contraction
			V_temp[seq_f] = V_front[seq]*W[index_start+j][seq_w];
			V_front = V_temp;
		}
		mttkrp_map[args] = V_front;
		return;
	}
	if(mttkrp_map.find(parent[args])==mttkrp_map.end()) {
		mttkrp_map_DT(mttkrp_map, parent, sibling, V, W, parent[args], dw);
	}
	/* Else
	*/
	Tensor<> V_temp;
	Tensor<> V_front = mttkrp_map[parent[args]];
	Tensor<> V_parent = mttkrp_map[parent[args]];
	int index_start = int(sibling[args][0]-parent[args][0]);
	// make seq_f
	char seq[V_front.order+1];
	char seq_f[V_front.order+1];
	seq_f[V_front.order] = '\0';
	seq_f[V_front.order-1] = '*';
	char_string_copy(seq_f,0,parent[args],0,parent[args].length());
	/* loops */
	for (int j=0; j<sibling[args].length(); j++) {     // iterate on [ab]
		// make seq
		seq[V_front.order] = '\0';
		strncpy(seq,seq_f, strlen(seq_f));
		// make seq_w
		seq_w[0] = sibling[args][j];
		// make seq_f
		seq_f[V_front.order-1] = '\0';
		seq_f[V_front.order-2] = '*';
		char_string_copy(seq_f,0,parent[args],0,index_start);
		char_string_copy(seq_f,index_start,parent[args],index_start+j+1,V_parent.order-index_start-j-2);
		// build len for V_temp
		int lens_V[V_parent.order-j-1];
		for (int m=0; m<index_start; m++) {
			lens_V[m] = V_parent.lens[m];
		}
		for (int m=index_start+j+1; m<V_parent.order; m++) {
			lens_V[m-j-1] = V_parent.lens[m];
		}
		lens_V[V_parent.order-j-2] = K;
		V_temp = Tensor<>(V_parent.order-j-1, lens_V, dw);
		// contraction
		V_temp[seq_f] = V_front[seq]*W[seq_w[0]-'a'][seq_w];
		V_front = V_temp;
	}
	mttkrp_map[args] = V_front;
	return;
}

void build_V(Tensor<> & V,
			 Matrix<> * W,
			 int order,
			 World & dw) {
  Timer tbuild_V("build_V");
  tbuild_V.start();
	char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
	// int lens_V[2];
	// lens_V[0] = W[0].nrow;
	// lens_V[1] = W[0].ncol;
	V = W[0];
	char seq_W[3] = {'i', '*', '\0'};
	// char seq = {'i','*','\0'};
	for (int i=1; i<order-1; i++) {
		// build V_temp
		int lens_V[i+2];
		for (int j=0; j<i+1; j++) {
			lens_V[j] = W[j].nrow;
		}
		lens_V[i+1] = W[0].ncol;
		Tensor<> V_temp = Tensor<>(i+2, lens_V, dw);
		// seq_temp
		char seq_temp[i+3];
		seq_temp[i+2] = '\0';
		seq_temp[i+1] = '*';
		for (int j=0; j<i+1; j++) {
			seq_temp[j] = chars[j];
		}
		// seq
		char seq[i+2];
		seq[i+1] = '\0';
		seq[i] = '*';
		for (int j=0; j<i; j++) {
			seq[j] = chars[j];
		}
		// seq_W
		seq_W[0] = chars[i];
		V_temp[seq_temp] = V[seq] * W[i][seq_W];
		V = V_temp;
		// char seq[i+3];
		// for (int j=0; j<i+3; j++) {
		// 	seq[j] = seq_temp[j];
		// }
	}
	// build V_temp
	int lens_V[order];
	for (int j=0; j<order; j++) {
		lens_V[j] = W[j].nrow;
	}
	Tensor<> V_temp = Tensor<>(order, lens_V, dw);
	char seq_temp[order+1];
	char seq[order+2];
	seq_temp[order] = '\0';
	seq_temp[order] = '\0';
	for (int j=0; j<order; j++) {
		seq_temp[j] = chars[j];
		seq[j] = chars[j];
	}
	seq[order-1] = '*';
	seq_W[0] = chars[order-1];

	V_temp[seq_temp] = V[seq] * W[order-1][seq_W];
	V = V_temp;
  tbuild_V.stop();

}

/**
 * \brief ALS method for CP decomposition with decision tree
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *	F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 *  V.order should be >=4
 */
bool alsCP_DT(Tensor<> & V,
			  Matrix<> * W,
			  Matrix<> * grad_W,
			  Matrix<> * F,
			  double tol,
			  double timelimit,
			  int maxiter,
			  double lambda,
        	  ofstream & Plot_File,
        	  int resprint,
        	  bool bench,
			  World & dw) {
	cout.precision(13);
	if (bench==false) {
    	if (dw.rank==0) Plot_File << "[dim],[iter],[gradnorm],[tol],[pp_update],[diffV],[dtime]" << "\n";          //Headings for file
	}

    Matrix<> regul =Matrix<>(W[0].ncol,W[0].ncol);
    regul["ii"] =  1.*lambda;

	double st_time = MPI_Wtime();
	int iter;
	double projnorm;
	double Fnorm=0.;
	double diffnorm_V = 1000;
	Matrix<> * grad_W_proj = new Matrix<>[V.order];
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
	// maps
	map<string, Tensor<>> mttkrp_map;
	map<string, string> parent;
	map<string, string> sibling;
	Construct_Dimension_Tree(parent, sibling, 0, V.order-1);
	for (iter=0; iter<=maxiter; iter++)
	{
		// print the gradient norm
		if (iter%resprint==0 || iter==maxiter) {
			double st_time1 = MPI_Wtime();
			//get the gradient
			// gradient_CP(V, W, grad_W, dw);
			// for (int i=0; i<V.order; i++) {
			// 	grad_W_proj[i] = Matrix<>(W[i].nrow,W[i].ncol);
			// 	grad_W_proj[i]["ij"] = grad_W[i]["ij"]-F[i]["ij"];
			// }
			projnorm = 0; //Fnorm = 0;
			for (int i=0; i<V.order; i++) {
				projnorm += grad_W[i].norm2()*grad_W[i].norm2();//grad_W_proj[i].norm2()*grad_W_proj[i].norm2();
				//Fnorm += F[i].norm2();
			}
			projnorm = sqrt(projnorm);
			// diffnorm
			Tensor<> V_build;
			build_V(V_build, W, V.order, dw);
			Tensor<> diff_V = V;
			diff_V[seq_V] = V[seq_V] - V_build[seq_V];
			diffnorm_V = diff_V.norm2();
			// record time
			st_time += MPI_Wtime() - st_time1;
			double dtime = MPI_Wtime() - st_time;
			if (bench==false) {
				if(dw.rank==0) {
					cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [gradnorm]  "<< projnorm << "  [tol]  " << tol << "  [pp_update]  " << 0  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
					Plot_File << V.lens[0] << "," << iter << "," << projnorm << "," << tol << "," << 0 << "," << diffnorm_V << "," << dtime << "\n";
					if(iter%100==0 && iter!=0) {// flush
						Plot_File << endl;
					}
				}
			} else {
				if(dw.rank==0 && iter!=0) {
					cout << "  [dimension tree step time]  " << dtime <<  "\n";
					Plot_File << "[DTtime]" << "," << dtime << "\n";
				}
			}
			// end check the residue
			if ((projnorm < tol) || MPI_Wtime()-st_time > timelimit)
				break;
		}
		// clear the Hash Table
		mttkrp_map.clear();
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
			/* initialize matrix M
			*/
			// make args
			char args[2];
			args[1] = '\0';
			args[0] = i+'a';
			if (mttkrp_map.find(parent[args])==mttkrp_map.end()) {
				mttkrp_map_DT(mttkrp_map, parent, sibling, V , W, parent[args], dw);
			}
			Matrix<> M = Matrix<>(W[i].nrow,W[i].ncol);
			if (sibling[args].length()==1) {
				char seq[3],seq_A[3],seq_p[4];
				seq[2] = '\0'; seq_A[2] = '\0'; seq_p[3] = '\0';
				seq[1] = '*'; seq_A[1] = '*'; seq_p[2] = '*';
				seq[0] = args[0]; seq_p[0] = parent[args][0]; seq_p[1] = parent[args][1];
				if (seq_p[0]==seq[0]) seq_A[0] = seq_p[1];
				else seq_A[0] = seq_p[0];
				M[seq] = mttkrp_map[parent[args]][seq_p]*W[int(seq_A[0]-'a')][seq_A];
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
				M[seq] = mttkrp_map[parent[args]][seq_p]*W[int(seq_A1[0]-'a')][seq_A1]*W[int(seq_A2[0]-'a')][seq_A2];
			}
			// Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
			// KhatriRao_contract(M2, V, W, index, lens_H, dw);
			// calculating S
			S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
			for (int ii=1; ii<V.order-1; ii++) {
				S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
			}
			S["ij"] += regul["ij"];
			// subproblem M=W*S
			M["ij"] += F[i]["ij"];
			// calculate gradient
			grad_W[i]["ij"] = -M["ij"]+W[i]["ik"]*S["kj"];
			SVD_solve(M, W[i], S);
			// recover the char
			temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;
		}
		Normalize(W, V.order, dw);
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final proj-grad norm %E \n", iter, projnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	if (bench==false) {
		Plot_File.close();
	}
	delete[] grad_W_proj;
	if (iter == maxiter+1) return false;
	else return true;
}



// [cd] --> [ab*]
void stringbuilder_mttkrp(char* seq,
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

void Build_mttkrp_map(map<string, Tensor<>> & mttkrp_map,
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
 * \brief ALS method for CP decomposition with dimension tree PP subroutine
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_DT_sub(Tensor<> & V,
					  Matrix<> * W,
        	  		  Matrix<> * grad_W,
					  Matrix<> * dW,
					  Matrix<> * F,
					  double tol,
					  double tol_init,
					  double timelimit,
					  int maxiter,
					  double & st_time,
					  double lambda,
					  ofstream & Plot_File,
					  double & projnorm,
					  int & iter,
					  int resprint,
					  World & dw) {

    Matrix<> regul =Matrix<>(W[0].ncol,W[0].ncol);
    regul["ii"] =  1.*lambda;

	// work as the preconditioning of pairwise perturbation
	Matrix<> W_prev[V.order];
	for (int i=0; i<V.order; i++) {
		W_prev[i] = Matrix<>(W[i].nrow,W[i].ncol);
	}

	double Fnorm = 0.;
	double diffnorm_V=1000;
	Matrix<> grad_W_proj[V.order];
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
	// maps
	map<string, Tensor<>> mttkrp_map;
	map<string, string> parent;
	map<string, string> sibling;
	Construct_Dimension_Tree(parent, sibling, 0, V.order-1);

	for (; iter<=maxiter; iter++)
	{
		// print the gradient norm
		if (iter%resprint==0 || iter==maxiter) {
			double st_time1 = MPI_Wtime();
			//get the gradient
			// gradient_CP(V, W, grad_W, dw);
			// for (int i=0; i<V.order; i++) {
			// 	grad_W_proj[i] = Matrix<>(W[i].nrow,W[i].ncol);
			// 	grad_W_proj[i]["ij"] = grad_W[i]["ij"]-F[i]["ij"];
			// }
			projnorm = 0; Fnorm = 0;
			for (int i=0; i<V.order; i++) {
				projnorm += grad_W[i].norm2()*grad_W[i].norm2();//grad_W_proj[i].norm2()*grad_W_proj[i].norm2();
				// Fnorm += F[i].norm2();
			}
			projnorm = sqrt(projnorm);
			// diffnorm
			Tensor<> V_build;
			build_V(V_build, W, V.order, dw);
			Tensor<> diff_V = V;
			diff_V[seq_V] = V[seq_V] - V_build[seq_V];
			diffnorm_V = diff_V.norm2();
			// record time
			st_time += MPI_Wtime() - st_time1;
			double dtime = MPI_Wtime() - st_time;
			if(dw.rank==0) {
				cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [gradnorm]  "<< projnorm << "  [tol]  " << tol << "  [pp_update]  " << 0  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
				// plot to file
				Plot_File << V.lens[0] << "," << iter << "," << projnorm << "," << tol << "," << 0 << "," << diffnorm_V << "," << dtime << "\n";
				if(iter%100==0 && iter!=0) {// flush
					Plot_File << endl;
				}
			}
			// end check the residue
			if ((projnorm < tol) || MPI_Wtime()-st_time > timelimit)
				break;
		}
		// clear the Hash Table
		mttkrp_map.clear();
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
			/* initialize matrix M
			*/
			// make args
			char args[2];
			args[1] = '\0';
			args[0] = i+'a';
			if (mttkrp_map.find(parent[args])==mttkrp_map.end()) {
				mttkrp_map_DT(mttkrp_map, parent, sibling, V , W, parent[args], dw);
			}
			Matrix<> M = Matrix<>(W[i].nrow,W[i].ncol);
			if (sibling[args].length()==1) {
				char seq[3],seq_A[3],seq_p[4];
				seq[2] = '\0'; seq_A[2] = '\0'; seq_p[3] = '\0';
				seq[1] = '*'; seq_A[1] = '*'; seq_p[2] = '*';
				seq[0] = args[0]; seq_p[0] = parent[args][0]; seq_p[1] = parent[args][1];
				if (seq_p[0]==seq[0]) seq_A[0] = seq_p[1];
				else seq_A[0] = seq_p[0];
				M[seq] = mttkrp_map[parent[args]][seq_p]*W[int(seq_A[0]-'a')][seq_A];
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
				M[seq] = mttkrp_map[parent[args]][seq_p]*W[int(seq_A1[0]-'a')][seq_A1]*W[int(seq_A2[0]-'a')][seq_A2];
			}
			// Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
			// KhatriRao_contract(M2, V, W, index, lens_H, dw);
			// calculating S
			S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
			for (int ii=1; ii<V.order-1; ii++) {
				S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
			}
			if (lambda!=0){
				S["ij"] += regul["ij"];
			}
			// subproblem M=W*S
			// M["ij"] += F[i]["ij"];
			grad_W[i]["ij"] = -M["ij"]+W[i]["ik"]*S["kj"];
			SVD_solve(M, W[i], S);
			// double norm_middle = W[i].norm2();
			// if (dw.rank==0) cout << norm_middle << endl;
			// recover the char
			temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;

		}
		if (Fnorm == 0) Normalize(W, V.order, dw);
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
		if (num_dw_break==V.order) return diffnorm_V;
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}

	return diffnorm_V;
}

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
double alsCP_PP_sub(Tensor<> & V,
				  Matrix<> * W,
        	  	  Matrix<> * grad_W,
				  Matrix<> * dW,
				  Matrix<> * F,
				  double tol,
				  double tol_init,
				  double timelimit,
				  int maxiter,
				  double & st_time,
				  double lambda,
				  double ratio_step,
				  ofstream & Plot_File,
				  double & projnorm,
				  int & iter,
				  int resprint,
				  bool bench,
				  World & dw){

	double dtime_first = 0;

	int init_iter = iter;

    Matrix<> regul =Matrix<>(W[0].ncol,W[0].ncol);
    regul["ii"] =  1.*lambda;

	double Fnorm = 0.;
	double diffnorm_V=1000;
	Matrix<> grad_W_proj[V.order];
	// initialize the dW matrices
	Matrix<> W_init[V.order];
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

	for (; iter<=maxiter; iter++)
	{
		// work as the preconditioning of pairwise perturbation
		int num_dw_break = 0;
		if (bench==false){
			for (int i=0; i<V.order; i++) {
				double norm_dW = dW[i].norm2();
				double norm_W = W[i].norm2();
				if (abs(norm_dW/norm_W)>tol_init) num_dw_break++;
			}
		}
		// initialize the MTTKRP
		if ((iter - init_iter)%15 == 0 || (num_dw_break > 0 ) ) {

			if (num_dw_break>0 || iter!=init_iter ) {
				return diffnorm_V;
			}
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
				Build_mttkrp_map(mttkrp_map, V, W, seq_tensor, dw);
			}
		}
		// print the gradient norm
		if (iter%resprint==0 || iter==maxiter || iter==init_iter) {
			double st_time1 = MPI_Wtime();
			//get the gradient
			// gradient_CP(V, W, grad_W, dw);
			// for (int i=0; i<V.order; i++) {
			// 	grad_W_proj[i] = Matrix<>(W[i].nrow,W[i].ncol);
			// 	grad_W_proj[i]["ij"] = grad_W[i]["ij"]-F[i]["ij"];
			// }
			projnorm = 0; //Fnorm = 0;
			for (int i=0; i<V.order; i++) {
				projnorm += grad_W[i].norm2()*grad_W[i].norm2(); //grad_W_proj[i].norm2()*grad_W_proj[i].norm2();
				// Fnorm += F[i].norm2();
			}
			projnorm = sqrt(projnorm);
			// diffnorm
			Tensor<> V_build;
			build_V(V_build, W, V.order, dw);
			Tensor<> diff_V = V;
			diff_V[seq_V] = V[seq_V] - V_build[seq_V];
			diffnorm_V = diff_V.norm2();
			// record time
			st_time += MPI_Wtime() - st_time1;
			double dtime = MPI_Wtime() - st_time;
			if (bench==false) {
				if(dw.rank==0) {
					cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [gradnorm]  "<< projnorm << "  [tol]  " << tol << "  [pp_update]  " << 1  << "  [diffV]  "  << diffnorm_V << "  [dtime]  " << dtime <<  "\n";
					// plot to file
					Plot_File << V.lens[0] << "," << iter << "," << projnorm << "," << tol << "," << 1 << "," << diffnorm_V << "," << dtime << "\n";
					if(iter%100==0 && iter!=0) {// flush
						Plot_File << endl;
					}
				}
			} else {
				if(dw.rank==0 && iter != maxiter) {
					dtime_first = dtime;
					st_time = MPI_Wtime();
				}
				else if (dw.rank==0 && iter == maxiter) {
					dtime_first = dtime_first+dtime;
					cout << "  [PP first time]  " << dtime_first <<  "\n";
					Plot_File << "  [PPfirst]  " << "," << dtime_first << "\n";
					cout << "  [PP second time]  " << dtime <<  "\n";
					Plot_File << "  [PPsecond]  " << "," << dtime << "\n";
				}
			}
			// end check the residue
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
			// KhatriRao_contract(M, V, W, index, lens_H, dw);
			char seq_M[V.order];
			seq_M[V.order-1] = '\0';
			strncpy(seq_M,seq,i);
			strncpy(seq_M+i,seq+i+1,V.order-i-1);
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
			if (lambda!=0) {
				S["ij"] += regul["ij"];
			}
			// // calculating S
			// S["ij"] = W_init[index[0]]["ki"]*W_init[index[0]]["kj"];
			// for (int ii=1; ii<V.order-1; ii++) {
			// 	S["ij"] = S["ij"]*(W_init[index[ii]]["ki"]*W_init[index[ii]]["kj"]);
			// }
			// subproblem M=W*S
			// M["ij"] += F[i]["ij"];
			grad_W[i]["ij"] = -M["ij"]+W[i]["ik"]*S["kj"];
			SVD_solve_mod(M, W[i], W_init[i], dW[i], S, ratio_step);
			// recover the char
			temp = seq_V[V.order-1];
			seq_V[V.order-1] = seq_V[i];
			seq_V[i] = temp;

			// if (Fnorm == 0) Normalize(W, V.order, dw);
			// double W_norm = W[i].norm2();
			// if (dw.rank==0) cout << W_norm << endl;
			// W[i]["ij"] = 1./W_norm*W[i]["ij"];
			// dW[i]["ij"] = W[i]["ij"] - W_init[i]["ij"];
		}
		if (Fnorm == 0) Normalize(W, V.order, dw);
		// print .
		if (iter%10==0 && dw.rank==0) printf(".");
	}
	if (bench==true) iter++;
	return diffnorm_V;
}

/**
 * \brief ALS method for CP decomposition
 *  W: output matrices
 *  V: input tensor
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_PP(Tensor<> & V,
        	  Matrix<> * W,
        	  Matrix<> * grad_W,
        	  Matrix<> * F,
        	  double tol,
        	  double tol_init,
        	  double timelimit,
        	  int maxiter,
        	  double lambda,
        	  double ratio_step,
          	  ofstream & Plot_File,
			  int resprint,
			  bool bench,
          	  World & dw) {
	cout.precision(13);

	if (bench==false){
		if (dw.rank==0) Plot_File << "[dim],[iter],[gradnorm],[tol],[pp_update],[diffV],[dtime]" << "\n";          //Headings for file
	}

	double st_time = MPI_Wtime();
	int iter = 0;
	double gradnorm = 10.;
	double diffnorm_V = 1.;
	// initialize dW
	Matrix<> * dW = new Matrix<>[V.order];
	for (int j=0; j<V.order; j++) {
		dW[j] = Matrix<>(W[j].nrow,W[j].ncol);
		dW[j]["ij"] = 0.;
	}

	while (gradnorm > tol && iter<=maxiter)
	{

		if (bench==false) {

			if (dw.rank==0) printf("DT starts from %d\n", iter);

			diffnorm_V = alsCP_DT_sub(V, W, grad_W, dW, F,
						 tol, tol_init,
						 timelimit, maxiter,
						 st_time, lambda, Plot_File,
						 gradnorm, iter, resprint, dw);

		}

		if (dw.rank==0) printf("pairwise perturbation starts from %d\n", iter);

		diffnorm_V = alsCP_PP_sub(V, W, grad_W, dW, F,
				    tol, tol_init,
					timelimit, maxiter,
					st_time, lambda, ratio_step, Plot_File,
					gradnorm, iter, resprint, bench, dw);
		// tol_init *= 0.9;

	}
	if(dw.rank==0) {
		printf ("\nIter = %d Final grad norm %E \n", iter, gradnorm);
		printf ("tf took %lf seconds\n",MPI_Wtime()-st_time);
	}
	if (bench==false){
		Plot_File.close();
	}
	delete[] dW;
	if (iter == maxiter+1) return false;
	else return true;
}



/**
 * \brief ALS method for CP decomposition with decision tree and rank1-update acceleration
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 *  V.order should be >=4
 *	return whether the ALS steps have converged.
 */
bool alsCP_rank1(Tensor<> & V,
 		   Matrix<> * W,
 		   Matrix<> * grad_W,
 		   double tol,
			 double tol_rank1,
 		   double timelimit,
 		   int maxiter,
 		   World & dw) {

	bool exceedsMaxTime = false;
	double start_time = MPI_Wtime();
	double gradnorm = tol+1;
	int iter = 0;
	//make the char
	char seq[V.order+1], seq_V[V.order+1];
	seq[V.order] = '\0'; seq_V[V.order] = '\0';
	for (int j=0; j<V.order; j++) {
		seq[j] = 'a'+j;
		seq_V[j] = seq[j];
	}

	// Gamma[n] = S[1]*S[2]*...*S[n-1]*S[n+1]*...*S[V.order], where S[i] = W[i].T.dot(W[i])
	// Save all S[i] since S[i] can be fast updated if we have rank1 updates on W[i]
	// Notice that the structure of CP allows us to save partial result of product of S into dimension tree
	Matrix<>* S = new Matrix<>[V.order];
	for (int i = 0; i<V.order; i++){S[i] = Matrix<>(W[0].ncol, W[0].ncol);}
	for (int i = 1; i<V.order; i++){S[i]["ij"] = W[i]["ki"]*W[i]["kj"];}

	unordered_map<string, Tensor<>>mttkrp_map;
	unordered_map<string, Matrix<>>gamma_map;
	unordered_map<string, string>parent;
	build_BDT(parent, seq, 0, V.order-1);
	fill_gamma_tree(gamma_map, S, seq, 0, V.order-1);
	mttkrp_map[seq] = V;

	// define two cached tensor
	// special index is 0 and V.order/2+1
	int tensor1_len[V.order];
	int tensor2_len[V.order];
	for (int i=0; i<V.order; i++){
		tensor1_len[i] = V.lens[i];
		tensor2_len[i] = V.lens[i];
	}
	tensor1_len[0] = W[0].ncol;
	tensor2_len[V.order/2+1] = W[0].ncol;
	Tensor<>* cached_tensor1 = new Tensor<>(V.order, tensor1_len, dw);
	Tensor<>* cached_tensor2 = new Tensor<>(V.order, tensor2_len, dw);

	update_cached_tensor(V, W, cached_tensor1, seq, 0);
	update_cached_tensor(V, W, cached_tensor2, seq, V.order/2+1);

	build_1st_level(mttkrp_map, V, W, cached_tensor1, cached_tensor2, dw);
	fill_mttkrp_tree(mttkrp_map, W, seq, 0, V.order/2, dw);
	fill_mttkrp_tree(mttkrp_map, W, seq, V.order/2+1, V.order-1, dw);

	double tempnorm;
	tuple<int, int> interval;
	int start, end;
	Tensor<> res_tensor;
	Matrix<> gamma, M, A_old;
	// Perform ALS for CP with dimension tree. The loop will be broke out if maximum iterations or maximum time is exceeded.
	while (iter<maxiter && gradnorm>tol){
		gradnorm = 0;

		for (int i = 0; i<V.order; i++){
			// first compute the mttkrp and S via dimension tree
			A_old = W[i]; // to compute G
			interval = find_interval(i, 0, V.order-1);
			start = get<0>(interval); end = get<1>(interval);
			char name[4]; name[2] = '\0'; name[3]='\0';
			for (int i=start; i<=end; i++){name[i-start]=seq[i];}
			res_tensor = mttkrp_map[name];
			gamma = gamma_map[name];
			M = Matrix<>(W[0].nrow, W[0].ncol);
			compute_gamma(gamma, W, i, start, end);
			compute_M(M, res_tensor, W, i, start, end, dw);

			SVD_solve(M, W[i], gamma);

			// If i is one of the special index, we need to update the cached_tensors
			if (i==0){
				update_cached_tensor(V, W, cached_tensor1, seq, 0);
				build_right_child(mttkrp_map, V, W, cached_tensor1, dw);
			}
			else if (i==V.order/2+1) {
				update_cached_tensor(V,W, cached_tensor2, seq, V.order/2+1);
				build_left_child(mttkrp_map, V, W, cached_tensor2, dw);
			}
			update_mttkrp_tree(mttkrp_map, W, seq, i, 0, V.order-1);
			update_gamma_tree(gamma_map, S, seq, i, 0, V.order-1);
			tempnorm = grad_W[i].norm2(); // gradient 2-norm squared
			gradnorm += tempnorm*tempnorm;
		}
		gradnorm = sqrt(gradnorm);
		iter++;
		if (MPI_Wtime()-start_time > timelimit) {exceedsMaxTime = true; break;}
	}
	delete[] S;
	delete cached_tensor1;
	delete cached_tensor2;
	if (iter==maxiter || exceedsMaxTime) return false;
	else return true;
}

/** Recursively construct the binary dimension tree (BDT)
	*/
void build_BDT(unordered_map<string, string> &parent_map, char* seq, int start, int end){
	if (end==start+1 || end==start+2) return;
	int mid = (start+end)/2;
	int child1_len = mid-start+1;
	char child1[child1_len+1];
	int child2_len = end-mid;
	char child2[child2_len+1]; // end-(mid+1)+2
	int parent_len = end-start+1;
	char parent[parent_len+1];
	for (int i = 0; i<parent_len; i++){
		parent[i] = seq[start+i];
	}
	parent[end-start+1] = '\0';
	for (int i = 0; i<child1_len; i++){
		child1[i] = seq[start+i];
	}
	child1[child1_len+1] = '\0';
	for (int i = 0; i<child2_len; i++){
		child2[i] = seq[mid+1+i];
	}
	child2[child2_len] = '\0';
	parent_map[child1] = parent;
	parent_map[child2] = parent;
	build_BDT(parent_map, seq, start, mid);
	build_BDT(parent_map, seq, mid+1, end);
}

void build_1st_level(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor1, Tensor<> *cached_tensor2, World &dw){
	build_left_child(mttkrp_map, V, W, cached_tensor2, dw);
	build_right_child(mttkrp_map, V, W, cached_tensor1, dw);
}

void build_left_child(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor2, World &dw){
	// compute the first level tree nodes
	// first compute M(1,2) (if the original tensor is M(1,2,3,4)) which is M(1,2,3,4) *W[3] *W[4]
	// This is the easier of the two because there is no change of order of modes.
	// notice the result (M(1,2)) will be a 3-d tensor
	int child1_name_len = V.order/2+1;
	char child1_name[child1_name_len+1]; child1_name[child1_name_len]='\0';
	for (int i=0; i<child1_name_len; i++) {child1_name[i] = 'a'+i;}

	int ndim = V.order-1;
	int len[V.order];
	for (int i=0; i<cached_tensor2->order; i++){len[i] = cached_tensor2->lens[i];}
	Tensor<> prev = *cached_tensor2;
	char seq_temp[V.order+1]; seq_temp[V.order]='\0';
	char seq_prev[V.order+1]; seq_prev[V.order]='\0';
	for (int i=0; i<prev.order; i++) {seq_prev[i] = 'a'+i; seq_temp[i] = seq_prev[i];}
	seq_temp[V.order-1]='\0';

	char seq_w[3]; seq_w[2]='\0'; seq_w[1] = seq_prev[child1_name_len];
	Tensor<> temp;
	for (int i=V.order-1; i>V.order/2+1; i--){
		seq_w[0] = seq_prev[i];
		temp = Tensor<>(ndim, len, dw);
		temp[seq_temp] = (prev)[seq_prev]*W[i][seq_w];
		prev = temp;
		seq_prev[i] = '\0';
		seq_temp[i-1] = '\0';
		ndim--;
	}
	mttkrp_map[child1_name] = prev;
}

void build_right_child(unordered_map<string, Tensor<>> &mttkrp_map, Tensor<> &V, Matrix<>*W, Tensor<> *cached_tensor1, World &dw){
	int child1_name_len = V.order/2+1;
	int child2_name_len = V.order - V.order/2-1; //V.order-1- (V.order/2+1)+1;
	char child2_name[child2_name_len+1]; child2_name[child2_name_len]='\0';
	for (int i=0; i<child2_name_len; i++) {child2_name[i] = 'a'+i+child1_name_len;}

	//compute M(3,4) Slightly more complicated than M(1,2) since it involves change of modes.
	int ndim = V.order;
	int len[ndim];
	char seq_prev[ndim+1]; seq_prev[ndim]='\0';
	for (int i=0; i<child2_name_len; i++){
		len[i] = cached_tensor1->lens[i+child1_name_len];
		seq_prev[i] = 'a'+i+child1_name_len;
	}
	for (int i=0; i<child1_name_len; i++){
		len[i+child2_name_len] = cached_tensor1->lens[i];
		seq_prev[i+child2_name_len] = 'a'+i;
	}
	char seq_temp[ndim];
	for (int i=0; i<V.order; i++) {seq_temp[i] = 'a'+i;}
	Tensor<> prev = Tensor<>(ndim, len, dw);
	prev[seq_prev] = (*cached_tensor1)[seq_temp];
	ndim--;

	for (int i=0; i<prev.order; i++) {seq_prev[i] = 'a'+i; seq_temp[i] = seq_prev[i];}
	seq_temp[V.order-1]='\0';

	Tensor<> temp;
	char seq_w[3]; seq_w[1] = seq_prev[child1_name_len]; seq_w[2]='\0';
	for (int i=V.order/2; i>0; i--){
		seq_w[0] = seq_prev[i+child2_name_len];
		temp = Tensor<>(ndim, len, dw);
		temp[seq_temp] = (prev)[seq_prev]*W[i][seq_w];
		prev = temp;
		seq_prev[i+child2_name_len] = '\0';
		seq_temp[i-1+child2_name_len] = '\0';
		ndim--;
	}
	mttkrp_map[child2_name] = prev;
	//char seq_temp[V.order+1]; seq_temp[V.order]='\0';
	//char seq_prev[V.order+1]; seq_prev[V.order]='\0';
	//for (int i=0; i<prev.order; i++) {seq_prev[i] = 'a'+i; seq_temp[i] = seq_prev[i];}
	//seq_temp[V.order-1]='\0';

}

void update_cached_tensor(Tensor<> &V, Matrix<> *W, Tensor<>* cached_tensor, char* seq, int i){
		char seq_V1[V.order+1]; seq_V1[V.order] = '\0';
		char seq_M[3]; seq_M[0] = seq[i]; seq_M[1] = 'a'+V.order; seq_M[2] = '\0';
		for (int j=0; j<V.order; j++) {
			seq_V1[j] = 'a'+j;
		}
		seq_V1[i] = 'a'+V.order;
		(*cached_tensor)[seq_V1] = V[seq]*W[0][seq_M];
}

/** Fill in the mttkrp map. Starting from the 1st level down.
*/

void fill_mttkrp_tree(unordered_map<string, Tensor<>>mttkrp_map, Matrix<> *W, char *seq, int start, int end, World &dw){
	if (end==start+1 || end==start+2) return;
	int mid = (start+end)/2;
	int child1_len = mid-start+1;
	char child1[child1_len+1];
	int child2_len = end-mid;
	char child2[child2_len+1]; // end-(mid+1)+2
	int parent_len = end-start+1;
	char parent[parent_len+1];
	for (int i = 0; i<parent_len; i++){
		parent[i] = seq[start+i];
	}
	parent[end-start+1] = '\0';
	for (int i = 0; i<child1_len; i++){
		child1[i] = seq[start+i];
	}
	child1[child1_len+1] = '\0';
	for (int i = 0; i<child2_len; i++){
		child2[i] = seq[mid+1+i];
	}
	child2[child2_len] = '\0';

	Tensor<> prev = mttkrp_map[parent];
	// compute M(3,4) i.e. the second child
	char seq2[prev.order+1]; seq2[prev.order]='\0';
	for (int i=0; i<prev.order; i++){seq2[i]='a'+i;}
	char seq_w[3]; seq_w[2]='\0'; seq_w[1] = seq2[prev.order-1];
	int len[prev.order]; for (int i=0; i<prev.order; i++) len[i]=prev.lens[i];
	int ndim = prev.order-1;
	for (int i=start;i<=mid; i++){
		seq_w[0]= seq2[i-start];
		Tensor<> temp = Tensor<>(ndim, len+1+i-start, dw);
		temp[seq2+1+i-start] = prev[seq2+i-start]*W[i][seq_w];
		prev = temp;
		ndim--;
	}
	mttkrp_map[child2] = prev;

	//compute M(1,2), the first child
	prev = mttkrp_map[parent];
	ndim = prev.order-1;
	char seq3[prev.order+1]; seq3[prev.order]='\0';
	for (int i=0; i<prev.order; i++) {seq3[i]='a'+i;}
	int pos = prev.order-1-1; // position of the last mode
	int last_len = len[prev.order-1];
	char last_char = 'a'+prev.order;
	seq_w[1] = last_char;
	for (int i=end; i>mid; i--){
		seq3[pos]=last_char; seq3[pos+1]='\0';
		seq2[pos-1] = last_char; seq3[pos]='\0';
		len[pos] = last_len;
		Tensor<> temp = Tensor<>(ndim, len, dw);
		temp[seq2] = prev[seq3]*W[i][seq_w];
		prev = temp;
		ndim--;
		pos--;
	}
	mttkrp_map[child1] = prev;
}

void fill_gamma_tree(unordered_map<string, Matrix<>> &gamma_map, Matrix<> *S, char* seq, int start, int end){
	if (end==start+1 || end==start+2) return;
	int mid = (start+end)/2;
	int child1_len = mid-start+1;
	char child1[child1_len+1];
	int child2_len = end-mid;
	char child2[child2_len+1]; // end-(mid+1)+2
	int parent_len = end-start+1;
	char parent[parent_len+1];
	for (int i = 0; i<parent_len; i++){
		parent[i] = seq[start+i];
	}
	parent[end-start+1] = '\0';
	for (int i = 0; i<child1_len; i++){
		child1[i] = seq[start+i];
	}
	child1[child1_len+1] = '\0';
	for (int i = 0; i<child2_len; i++){
		child2[i] = seq[mid+1+i];
	}
	child2[child2_len] = '\0';

	Matrix<> temp = gamma_map[parent];
	// child1
	for (int i = mid+1; i<=end; i++){temp["ij"] = temp["ij"]*S[i]["ij"];}
	gamma_map[child1] = temp;

	temp = gamma_map[parent];
	for (int i=start; i<=mid; i++) temp["ij"] = temp["ij"]*S[i]["ij"];
	gamma_map[child2] = temp;
	fill_gamma_tree(gamma_map, S, seq, start, mid);
	fill_gamma_tree(gamma_map, S, seq, mid+1, end);
}

/** Update the dimension tree from the first level (not from the root) if one of the mode changes. Assume the
	* cached tensors and the first level tensors are already updated. For example, assume the original tensor is a 4-mode
	* tensor. Then the root is M(1,2,3,4) and its children are M(1,2) and M(3,4). The cached tensors are
	* M(2,3,4) and M(1,2,4) (i.e. the root contracted with mode[0] and mode[mid+1]).
	*/
void update_gamma_tree(unordered_map<string, Matrix<>> &gamma_map, Matrix<>* S, char* seq, int index, int start, int end){
	if (end==start+1 || end==start+2) return;
	int mid = (start+end)/2;
	int child1_len = mid-start+1;
	char child1[child1_len+1];
	int child2_len = end-mid;
	char child2[child2_len+1]; // end-(mid+1)+2
	int parent_len = end-start+1;
	char parent[parent_len+1];
	for (int i = 0; i<parent_len; i++){
		parent[i] = seq[start+i];
	}
	parent[end-start+1] = '\0';
	for (int i = 0; i<child1_len; i++){
		child1[i] = seq[start+i];
	}
	child1[child1_len+1] = '\0';
	for (int i = 0; i<child2_len; i++){
		child2[i] = seq[mid+1+i];
	}
	child2[child2_len] = '\0';

	if (index<start || index>mid){ // I need update child1
		Matrix<> temp = gamma_map[parent];
		// child1
		for (int i = mid+1; i<=end; i++){temp["ij"] = temp["ij"]*S[i]["ij"];}
		gamma_map[child1] = temp;
	}
	if (index<mid || index>end){// I need to update child2
		Matrix<> temp = gamma_map[parent];
		for (int i=start; i<=mid; i++) temp["ij"] = temp["ij"]*S[i]["ij"];
		gamma_map[child2] = temp;
	}
	update_gamma_tree(gamma_map, S, seq, index, start, mid);
	update_gamma_tree(gamma_map, S, seq, index, mid+1, end);
}

/** recursively find the leaf nodes. Assume start <= index <= end.
*/
tuple<int, int> find_interval(int index, int start, int end){
	if (start+1==end || start+2==end){
		return make_tuple(start, end);
	}
	int mid = (start+end)/2;
	if (start<=index && index<=mid) return find_interval(index, start, mid);
	else return find_interval(index, mid+1, end);
}

/** Compute the result gamma matrix. res is imported as a partial result.
	*/
void compute_gamma(Matrix<> &res, Matrix<> *W, int index, int start, int end){
	for (int i=start; i<=end; i++){
		if (i!=index){
			res["ij"] = W[i]["ij"]*res["ij"];
		}
	}
}

void compute_M(Matrix<> &M, Tensor<> &res, Matrix<> *W, int index, int start, int end, World &dw){
	if (start+1==end){
		//Tensor<> temp;
		if (index==start){
			//int len[] = {res.lens[1], res.lens[2]};
			//temp = Tensor<>(2, len, dw);
			M["ik"] = res["ijk"]*W[end]["jk"];
		}
		else {M["jk"] = res["ijk"]*W[start]["ik"];}
	}
	else if (start+2==end){
		Tensor<> temp;
		if (start==index){
			int len[] = {res.lens[0], res.lens[2], res.lens[3]};
			temp = Tensor<>(3, len, dw);
			temp["ikl"] = res["ijkl"]*W[start+1]["jl"];
			M["il"] = temp["ikl"]*W[end]["kl"];
		}
		else if (end==index){
			int len[] = {res.lens[1], res.lens[2], res.lens[3]};
			temp = Tensor<>(3, len, dw);
			temp["jkl"] = res["ijkl"]*W[start+1]["il"];
			M["kl"] = temp["jkl"]*W[end]["jl"];
		}
		else {
			int len[] = {res.lens[1], res.lens[2], res.lens[3]};
			temp = Tensor<>(3, len, dw);
			temp["jkl"] = res["ijkl"]*W[start+1]["il"];
			M["jl"] = temp["jkl"]*W[end]["kl"];
		}
	}
}
