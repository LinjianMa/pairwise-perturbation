
// #include "decomposition.h"
#include <ctf.hpp>
#include "../../common.h"


using namespace CTF;

template<typename dtype>
CPDTLROptimizer<dtype>::CPDTLROptimizer(int order, int r, int update_rank, int randomsvd, World & dw)
    : CPDTOptimizer<dtype>(order, r, dw){

    if (randomsvd>0) {
        this->randomsvd = true;
    } else {
        this->randomsvd = false;
    }

    num_subiteration = 5;
    rank = update_rank;
    cached_tensor1 = NULL;
    cached_tensor2 = NULL;
    initialize_low_rank_param();
}

template<typename dtype>
void CPDTLROptimizer<dtype>::initialize_low_rank_param() {
    count_subiteration = 0;
    low_rank_decomp = false;
}

template<typename dtype>
CPDTLROptimizer<dtype>::~CPDTLROptimizer(){
    // delete S;
    delete cached_tensor1;
    delete cached_tensor2;
}

template<typename dtype>
void CPDTLROptimizer<dtype>::mttkrp_map_init(int left_index) {
    World * dw = this->world;
    int order = this->order;

    char * seq_map_init = this->seq_map_init;
    char * seq_tree_top = this->seq_tree_top;
    char * seq_V = this->seq_V;

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
    for (int ii=0; ii<strlen(seq_map_init); ii++) {
        if (seq_map_init[ii] == '*') {
            lens[ii] = this->W[0].ncol;
        } else {
            lens[ii] = this->V->lens[int(seq_map_init[ii]-'a')];
        }
    }
    this->mttkrp_map[seq_tree_top] = Tensor<dtype>(strlen(seq_map_init), lens, *dw);
    if (this->low_rank_decomp && count_subiteration>1) {
        update_cached_tensor(left_index);
        if (this->first_subtree) {
            this->mttkrp_map[seq_tree_top] = *cached_tensor1;
        } else {
            this->mttkrp_map[seq_tree_top] = *cached_tensor2;
        }
    } else {
        this->mttkrp_map[seq_tree_top][seq_map_init] = (*this->V)[seq_V] * this->W[left_index][seq_matrix];
        if (this->first_subtree) {
            if (cached_tensor1==NULL) {
                cached_tensor1 = new Tensor<dtype>(strlen(seq_map_init), lens, *dw);
            }
            *cached_tensor1 = this->mttkrp_map[seq_tree_top];
        }
        else {
            if (cached_tensor2==NULL) {
                cached_tensor2 = new Tensor<dtype>(strlen(seq_map_init), lens, *dw);
            }
            *cached_tensor2 = this->mttkrp_map[seq_tree_top];
        }
    }
}

template<typename dtype>
void CPDTLROptimizer<dtype>::mttkrp_map_DT(string index) {
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

    for (int ii=0; ii<strlen(index_char); ii++) {
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
void CPDTLROptimizer<dtype>::update_cached_tensor(int left_index){

    char * seq_map_init = this->seq_map_init;
    char * seq_tree_top = this->seq_tree_top;
    char * seq_V = this->seq_V;

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

    if (this->first_subtree) {
        (*cached_tensor1)[seq_map_init] += (*this->V)[seq_V] * this->U[seq_U] * this->VT[seq_VT];
    }
    else {
        (*cached_tensor2)[seq_map_init] += (*this->V)[seq_V] * this->U[seq_U] * this->VT[seq_VT];
    }
}

template<typename dtype>
double CPDTLROptimizer<dtype>::step() {

    World * dw = this->world;
    int order = this->order;

    if (this->first_subtree) {
        this->indexes = this->indexes1;
        this->left_index = this->left_index1;
    }
    else {
        this->indexes = this->indexes2;
        this->left_index = this->left_index2;
    }
    // clear the Hash Table
    this->mttkrp_map.clear();
    // reinitialize
    //update_indexes();
    mttkrp_map_init(this->left_index);
    // iteration on W[i]
    for (int i=0; i<this->indexes.size(); i++) {

        if (this->first_subtree && i<this->special_index)
            continue;
        if (!this->first_subtree && i>this->special_index)
            break;
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
        if (((this->first_subtree && i==this->indexes.size()-1) || (!this->first_subtree && i==0)) && count_subiteration>=1) {
            get_rankR_update_cholesky(this->rank, this->U, this->s, this->VT, M, this->W[this->indexes[i]], this->S, this->randomsvd);
            this->W[this->indexes[i]]["ij"] += this->U["ik"]*this->s["k"]*this->VT["kj"];
            //update_cached_tensor(indexes[i]);
            this->low_rank_decomp = true;
        } else {
            cholesky_solve(M, this->W[this->indexes[i]], this->S);
        }
    }
    if (!this->first_subtree) {
        count_subiteration ++;
    }
    if (count_subiteration==num_subiteration && (!this->first_subtree)) {
        this->special_index = (this->special_index + 1)%(order-1);
        initialize_low_rank_param();
        if (this->special_index!=0) {
            this->left_index1 = (this->left_index1 + order - 1) % order;
            this->left_index2 = (this->left_index2 + order - 1) % order;
        } else {
            this->left_index = order-1;
            this->left_index1 = this->left_index;
            this->left_index2 = (this->left_index + order - 1) % order;
        }
        CPDTOptimizer<dtype>::update_indexes(this->indexes1, this->left_index1);
        CPDTOptimizer<dtype>::update_indexes(this->indexes2, this->left_index2);
    }
    this->first_subtree = !this->first_subtree;

    return 0.5;
}
