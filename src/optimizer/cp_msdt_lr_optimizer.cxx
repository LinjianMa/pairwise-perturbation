
// #include "decomposition.h"
#include <ctf.hpp>
#include "../../common.h"


using namespace CTF;

template<typename dtype>
CPMSDTLROptimizer<dtype>::CPMSDTLROptimizer(int order, int r, int update_rank, int randomsvd, World & dw)
    : CPMSDTOptimizer<dtype>(order, r, dw) {

    if (randomsvd > 0) {
        this->randomsvd = true;
    } else {
        this->randomsvd = false;
    }

    rank = update_rank;
    low_rank_decomp = false;
    is_cached = new bool[order];
    for (int i=0; i<order; i++){is_cached[i]=false;}
    cached_tensors = new Tensor<dtype>[order];
    old_W = new Matrix<dtype>[order];
}

template<typename dtype>
CPMSDTLROptimizer<dtype>::~CPMSDTLROptimizer(){
    // delete S;
    delete[] is_cached;
    delete[] cached_tensors;
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::mttkrp_map_init(int left_index) {
    World * dw = this->world;
    int order = this->order;

    char * seq_V = this->seq_V;
    char * seq_map_init = this->seq_map_init;
    char * seq_tree_top = this->seq_tree_top;

    // build seq_map_init
    seq_map_init[order] = '\0';
    seq_map_init[order-1] = '*';
    int j = 0;
    for (int i=left_index+1; i<order; i++){
        seq_map_init[j] = 'a' + i;
        j++;
    }
    for (int i=0; i<left_index; i++) {
        seq_map_init[j] = 'a' + i;
        j++;
    }
    // build seq_matrix
    char seq_matrix[3];
    seq_matrix[2] = '\0';
    seq_matrix[1] = '*';
    seq_matrix[0] = 'a'+left_index;
    // store that into the mttkrp_map
    int lens[strlen(seq_map_init)];
    for (int ii=0; ii<strlen(seq_map_init); ii++){
        if (seq_map_init[ii] == '*') {
            lens[ii] = this->W[0].ncol;
        } else {
            lens[ii] = this->V->lens[int(seq_map_init[ii]-'a')];
        }
    }
    this->mttkrp_map[seq_tree_top] = Tensor<dtype>(strlen(seq_map_init), lens, *dw);
    if (this->low_rank_decomp && this->is_cached[left_index]) {
        update_cached_tensor(left_index);
        this->mttkrp_map[seq_tree_top] = cached_tensors[left_index];
    } else {
        this->mttkrp_map[seq_tree_top][seq_map_init] = (*this->V)[seq_V] * this->W[left_index][seq_matrix];
        cached_tensors[left_index] = this->mttkrp_map[seq_tree_top];
        old_W[left_index] = this->W[left_index];
        this->is_cached[left_index] = true;
    }
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::mttkrp_map_DT(string index) {
    World * dw = this->world;
    char const * index_char = index.c_str();

    char const * parent_index = this->parent[index].c_str();
    if (this->mttkrp_map.find(parent_index)==this->mttkrp_map.end()) {
        mttkrp_map_DT(parent_index);
    }
    // get the modindexe of W
    char const * mat_index = this->contract_index[index].c_str();
    int W_index = int(mat_index[0] - 'a');
    int lens[strlen(index_char)];

    for (int ii=0; ii<strlen(index_char); ii++){
        if (index[ii] == '*') {
            lens[ii] = this->W[0].ncol;
        } else {
            lens[ii] = this->V->lens[int(this->indexes[index[ii]-'a'])];
        }
    }
    this->mttkrp_map[index] = Tensor<dtype>(strlen(index_char), lens, *dw);

    this->mttkrp_map[index][index_char] = this->mttkrp_map[parent_index][parent_index] * this->W[this->indexes[W_index]][mat_index];
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::update_cached_tensor(int left_index) {

    char * seq_V = this->seq_V;
    char * seq_map_init = this->seq_map_init;
    char * seq_tree_top = this->seq_tree_top;

    int order = this->order;
    seq_map_init[order] = '\0';
    seq_map_init[order-1] = '*';
    int j = 0;
    for (int i=left_index+1; i<order; i++){
        seq_map_init[j] = 'a' + i;
        j++;
    }
    for (int i=0; i<left_index; i++) {
        seq_map_init[j] = 'a' + i;
        j++;
    }
    this->U["ij"] = this->U["ij"]*this->s["j"];
    char seq_U[] = {char('a'+left_index), '&', '\0'};
    char seq_VT[] = {'&', '*','\0'};
    char seq_matrix[] = {char('a'+left_index), '*', '\0'};
    //cached_tensors[left_index][seq_map_init] = this->W[left_index][seq_matrix] * (*this->V)[seq_V];
    char seq_temp[order];
    int i = 0;
    while (seq_V[i]!='\0') {
        seq_temp[i] = seq_V[i];
        i++;
    }
    seq_temp[left_index] = '&';
    int lens[order];
    for (int i=0; i<order; i++) {
        lens[i] = this->V->lens[i];
    }
    lens[left_index] = rank;
    Tensor<dtype> temp = Tensor<dtype>(order, lens, *this->world);
    temp[seq_temp] = (*this->V)[seq_V] * this->U[seq_U];
    cached_tensors[left_index][seq_map_init] = cached_tensors[left_index][seq_map_init] + this->VT[seq_VT]*temp[seq_temp];

    old_W[left_index] = this->W[left_index];
    this->is_cached[left_index] = true;
}

template<typename dtype>
double CPMSDTLROptimizer<dtype>::step() {

    World * dw = this->world;
    int order = this->order;

    // clear the Hash Table
    this->mttkrp_map.clear();
    // reinitialize
    CPMSDTOptimizer<dtype>::update_indexes();
    // cout << left_index << endl;
    mttkrp_map_init(this->left_index);

    // iteration on W[i]
    for (int i=0; i<this->indexes.size(); i++) {
        /*  construct Matrix M
        *   M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
        */
        vector<int> mat_index = {i};

        string mat_seq;
        vec2str(mat_index, mat_seq);

        if (this->mttkrp_map.find(mat_seq)==this->mttkrp_map.end()) {
            mttkrp_map_DT(mat_seq);
        }
        Matrix<dtype> M = this->mttkrp_map[mat_seq];

        // calculating S
        CPOptimizer<dtype>::update_S(this->indexes[i]);
        // calculate gradient
        this->grad_W[this->indexes[i]]["ij"] = -M["ij"]+this->W[this->indexes[i]]["ik"]*this->S["kj"];
        if (!is_cached[this->indexes[i]] || (i!=(this->indexes.size()-1))){
            cholesky_solve(M, this->W[this->indexes[i]], this->S);
        } else {
            get_rankR_update_cholesky(this->rank, this->U, this->s, this->VT, M, this->old_W[this->indexes[i]], this->S, this->randomsvd);
            this->W[this->indexes[i]]["ij"] = this->old_W[this->indexes[i]]["ij"] + this->U["ik"]*this->s["k"]*this->VT["kj"];
            //update_cached_tensor(indexes[i]);
            this->low_rank_decomp = true;
        }
    }
    return 1.*(this->order-1)/this->order;
}
