/** \addtogroup examples 
  * @{ 
  * \defgroup als_tensor_factorization als_tensor_factorization
  * @{ 
  * \brief NTF algorithms based on projected gradient methods
  */
#include <ctf.hpp>
#include "common.cxx"
using namespace CTF;
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
	if (iter == maxiter+1) return false;
	else return true;
}

void Construct_Dimension_Tree(map<string, string>& parent,
							  map<string, string>& sibling, 
							  int start, 
							  int end) {
	if (end==start) return;
	if (end==start+1) {
		char args_parent[3];
		args_parent[2] = '\0';
		args_parent[1] = 'a'+end;
		args_parent[0] = 'a'+start;
		char args[2];
		char args2[2];
		args[1] = '\0'; args2[1] = '\0';
		args[0] = 'a'+start; args2[0] = 'a'+end;
		parent[args] = args_parent;
		parent[args2] = args_parent;
		sibling[args] = args2;
		sibling[args2] = args;
		return;
	}
	char args_parent[end-start+2];
	args_parent[end-start+1] = '\0';
	for (int i=start;i<=end;i++) {
		args_parent[i-start] = 'a'+i;
	}
	int middle = (start+end)/2;
	char args[middle-start+2];
	args[middle-start+1] = '\0';
	for (int i=start;i<=middle;i++) {
		args[i-start] = 'a'+i;
	}
	char args2[end-middle+1];
	args2[end-middle] = '\0';
	for (int i=middle+1;i<=end;i++) {
		args2[i-middle-1] = 'a'+i;
	}
	parent[args] = args_parent;
	sibling[args] = args2;
	Construct_Dimension_Tree(parent, sibling, start, middle);	
	sibling[args2] = args;
	parent[args2] = args_parent;
	Construct_Dimension_Tree(parent, sibling, middle+1, end);
	return;
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
		if (dw.rank==0) cout <<"shere1" << endl;
		if (dw.rank==0) cout <<seq_f << endl;
		if (dw.rank==0) cout <<seq << endl;
		if (dw.rank==0) cout <<seq_w << endl;
		if (dw.rank==0)cout <<"index" << index_start << endl;

		V_temp[seq_f] = V_front[seq]*W[index_start][seq_w];
		if (dw.rank==0) cout <<"shere2" << endl;
		V_front = V_temp;
		/* loops
		*/
		for (int j=1; j<sibling[args].length(); j++) {     // iterate on [ab]
			// make seq
			// char seq[V_front.order+1];
			seq[V_front.order] = '\0';
			strncpy(seq,seq_f, strlen(seq_f));
			// make seq_w
			seq_w[0] = sibling[args][j];
			// make seq_f
			// char seq_f[V_front.order];
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
		if (dw.rank==0) cout <<"shere11" << endl;
		if (dw.rank==0) cout <<seq_f << endl;
		if (dw.rank==0) cout <<seq << endl;
		if (dw.rank==0) cout <<seq_w << endl;
		if (dw.rank==0)cout <<"index" << index_start+j << endl;

			V_temp[seq_f] = V_front[seq]*W[index_start+j][seq_w];
		if (dw.rank==0) cout <<"shere12" << endl;
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
				if (dw.rank==0) cout <<seq_f << endl;
				if (dw.rank==0) cout <<V_parent.order-index_start-j-1 << endl;

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
				if (dw.rank==0) cout <<"ddhere11" << endl;
		if (dw.rank==0) cout <<seq_f << endl;
		if (dw.rank==0) cout <<seq << endl;
		if (dw.rank==0) cout <<seq_w << endl;
		if (dw.rank==0)cout <<"index" << index_start+j << endl;
		V_temp[seq_f] = V_front[seq]*W[seq_w[0]-'a'][seq_w];
						if (dw.rank==0) cout <<"ddhere22" << endl;

			// 	int lensT2[4] = {10,10,10,4};
			// Tensor<>T2 = Tensor<>(4,lensT2,dw);
			//  Tensor<>dT2 = Tensor<>(4,lensT2,dw);

			//  T2["abc*"]= V["abcdef"]*W[3]["d*"]*W[4]["e*"]*W[5]["f*"];
			//  dT2["abc*"] = T2["abc*"]- V_front["abc*"];
			//  double dT2_norm = dT2.norm2();
			//  if(dw.rank==0) cout <<dT2_norm << endl;	
			// 		int lensT2[4] = {10,10,4};
			// Tensor<>T2 = Tensor<>(3,lensT2,dw);
			//  Tensor<>dT2 = Tensor<>(3,lensT2,dw);
			//  			T2["ab*"]= V["abcdef"]*W[2]["c*"]*W[3]["d*"]*W[4]["e*"]*W[5]["f*"];
			//  dT2["ab*"] = T2["ab*"]- V_temp["ab*"];
			//  double dT2_norm = dT2.norm2();
			//  if(dw.rank==0) cout <<dT2_norm << endl;						

		V_front = V_temp;
	}
			// 	int lensT2[4] = {10,10,4};
			// Tensor<>T2 = Tensor<>(3,lensT2,dw);
			//  Tensor<>dT2 = Tensor<>(3,lensT2,dw);
			//  			T2["ab*"]= V["abcdef"]*W[2]["c*"]*W[3]["d*"]*W[4]["e*"]*W[5]["f*"];
			//  dT2["ab*"] = T2["ab*"]- V_front["ab*"];
			//  double dT2_norm = dT2.norm2();
			//  if(dw.rank==0) cout <<dT2_norm << endl;	
	mttkrp_map[args] = V_front;
	return;
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
	// maps 
	map<string, Tensor<>> mttkrp_map;
	map<string, string> parent;
	map<string, string> sibling;
	Construct_Dimension_Tree(parent, sibling, 0, V.order-1);

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
			if(dw.rank==0) cout << "  [dim]=  " << V.lens[0] << "  [iter]=  " << iter << "  [projnorm]  "<< projnorm << "  [tol]  " << tol << "  [Fnorm]  " << Fnorm <<  endl;
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
			if (dw.rank==0) cout << "here" << endl;
			if (mttkrp_map.find(parent[args])==mttkrp_map.end()) {
				mttkrp_map_DT(mttkrp_map, parent, sibling, V , W, parent[args], dw);
			}
			if (dw.rank==0) cout << "here2" << endl;
			Matrix<> M = Matrix<>(W[i].nrow,W[i].ncol);
			if (sibling[args].length()==1) {
				char seq[3],seq_A[3],seq_p[4];
				seq[2] = '\0'; seq_A[2] = '\0'; seq_p[3] = '\0';
				seq[1] = '*'; seq_A[1] = '*'; seq_p[2] = '*';
				seq[0] = args[0]; seq_p[0] = parent[args][0]; seq_p[1] = parent[args][1];
				if (seq_p[0]==seq[0]) seq_A[0] = seq_p[1];
				else seq_A[0] = seq_p[0];
				if (dw.rank==0) cout << seq << "  "<< seq_p << "  "<< seq_A << endl; 
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
				if (dw.rank==0) cout << seq << "  "<< seq_p << "  "<< seq_A1 << "  " << seq_A2 << endl; 
				M[seq] = mttkrp_map[parent[args]][seq_p]*W[int(seq_A1[0]-'a')][seq_A1]*W[int(seq_A2[0]-'a')][seq_A2];				
			}
						Matrix<> M2 = Matrix<>(W[i].nrow,W[i].ncol);
						Matrix<> dM2 = Matrix<>(W[i].nrow,W[i].ncol);
			// Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
			 KhatriRao_contract(M2, V, W, index, lens_H, dw);
						// M2["a*"] = V["abcdef"]*W[3]["d*"]*W[4]["e*"]*W[5]["f*"]*W[2]["c*"]*W[1]["b*"];
			 dM2["ij"] = M2["ij"]-M["ij"];
			 double dM_norm = dM2.norm2();
			 if(dw.rank==0) cout <<dM_norm << endl;
			 // int lensT2[5] = {10,10,4};
			 // Tensor<>T2 = Tensor<>(3,lensT2,dw);
			 // Tensor<>dT2 = Tensor<>(3,lensT2,dw);

			 // T2["ab*"]= V["abcdef"]*W[2]["c*"]*W[3]["d*"]*W[4]["e*"]*W[5]["f*"];
			 // dT2["ab*"] = T2["ab*"]- mttkrp_map["ab"]["ab*"];
			 // double dT2_norm = dT2.norm2();
			 // if(dw.rank==0) cout <<dT2_norm << endl;			 
			// calculating S
			S["ij"] = W[index[0]]["ki"]*W[index[0]]["kj"];
			for (int ii=1; ii<V.order-1; ii++) {
				S["ij"] = S["ij"]*(W[index[ii]]["ki"]*W[index[ii]]["kj"]);
			}
			// subproblem M=W*S
			M["ij"] += F[i]["ij"];
			SVD_solve(M, W[i], S);
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
 * \brief ALS method for CP decomposition
 *  W: output solutions
 *  V: input tensor
 *  grad_W: gradient in each dimension
 *	F: correction terms, F[]=0 initially
 *  tol: tolerance for a relative stopping condition
 *  timelimit, maxiter: limit of time and iterations
 */
bool alsCP_mod(Tensor<> & V, 
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
			seq_M[V.order-1] = '\0';
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