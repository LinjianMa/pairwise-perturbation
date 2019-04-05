
// #include "decomposition.h"
#include <ctf.hpp>
#include "../../common.h"


using namespace CTF;

template<typename dtype>
CPMSDTLROptimizer<dtype>::CPMSDTLROptimizer(int order, int r, int update_rank, World & dw)
    : CPOptimizer<dtype>(order, r, dw){

    // make the char seq_V
    seq_V[order] = '\0';
    seq_tree_top[order] = '\0';
    for (int j=0; j<order; j++) {
        seq_V[j] = 'a'+j;
        seq_tree_top[j] = seq_V[j];
    }
    seq_tree_top[order-1] = '*';

    // construct the tree
    Construct_Dimension_Tree();

    // initialize the indexes
    indexes = vector<int>(order-1, 0);
    for (int i=0; i<indexes.size(); i++) {
        indexes[i] = i;
    }
    left_index = order;

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
void CPMSDTLROptimizer<dtype>::update_indexes() {
    int order = this->order;
    left_index = (left_index + order - 1) % order;

    int j = 0;
    for (int i=left_index+1; i<order; i++) {
        indexes[j] = i;
        j++;
    }
    for (int i=0; i<left_index; i++) {
        indexes[j] = i;
        j++;
    }
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::vec2str(vector<int> vec, string & seq_out) {
    char seq[vec.size()+2];
    seq[vec.size()+1] = '\0';
    seq[vec.size()] = '*';
    for (int i=0; i<vec.size(); i++) {
        seq[i] = 'a' + vec[i];
    }
    seq_out = seq;
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::Construct_Dimension_Tree() {
    int order = this->order;
    vector<int> top_node = vector<int>(order-1);
    for (int i=0; i<top_node.size(); i++) {
        top_node[i] = i;
    }

    Construct_Subtree(top_node);
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::Construct_Subtree(vector<int> top_node) {
    Right_Subtree(top_node);

    vector<int> child_node = vector<int>(top_node.size()-1);
    for (int i=0; i<child_node.size(); i++) {
        child_node[i] = top_node[i];
    }

    vector<int> mat_index = {top_node[top_node.size()-1]};

    string child_seq, top_seq, mat_seq;
    vec2str(child_node, child_seq);
    vec2str(top_node, top_seq);
    vec2str(mat_index, mat_seq);

    parent[child_seq] = top_seq;
    contract_index[child_seq] = mat_seq;

    if (child_node.size()>1) {
        Construct_Subtree(child_node);
    }

}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::Right_Subtree(vector<int> top_node) {
    // construct the right tree
    vector<int> child_node = vector<int>(top_node.size()-1);
    for (int i=0; i<child_node.size(); i++) {
        child_node[i] = top_node[i];
    }
    child_node[child_node.size()-1] = top_node[top_node.size()-1];

    vector<int> mat_index = {top_node[top_node.size()-2]};

    string child_seq, top_seq, mat_seq;
    vec2str(child_node, child_seq);
    vec2str(top_node, top_seq);
    vec2str(mat_index, mat_seq);

    parent[child_seq] = top_seq;
    contract_index[child_seq] = mat_seq;

    if (child_node.size()>1) {
        Right_Subtree(child_node);
    }
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::mttkrp_map_init(int left_index) {
    World * dw = this->world;
    int order = this->order;

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
        if (seq_map_init[ii] == '*') lens[ii] = this->W[0].ncol;
        else lens[ii] = this->V->lens[int(seq_map_init[ii]-'a')];
    }
    mttkrp_map[seq_tree_top] = Tensor<dtype>(strlen(seq_map_init), lens, *dw);
    if (this->low_rank_decomp && this->is_cached[left_index]){
        update_cached_tensor(left_index);
        mttkrp_map[seq_tree_top] = cached_tensors[left_index];
    } else {
        mttkrp_map[seq_tree_top][seq_map_init] = (*this->V)[seq_V] * this->W[left_index][seq_matrix];
        cached_tensors[left_index] = mttkrp_map[seq_tree_top];
        old_W[left_index] = this->W[left_index];
        //cout<<"update old_W "<<left_index<<endl;
        this->is_cached[left_index] = true;
    }
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::mttkrp_map_DT(string index) {
    World * dw = this->world;
    char const * index_char = index.c_str();

    char const * parent_index = parent[index].c_str();
    if (mttkrp_map.find(parent_index)==mttkrp_map.end()) {
        mttkrp_map_DT(parent_index);
    }
    // get the modindexe of W
    char const * mat_index = contract_index[index].c_str();
    int W_index = int(mat_index[0] - 'a');
    int lens[strlen(index_char)];

    for (int ii=0; ii<strlen(index_char); ii++){
        if (index[ii] == '*') lens[ii] = this->W[0].ncol;
        else lens[ii] = this->V->lens[int(indexes[index[ii]-'a'])];
    }
    mttkrp_map[index] = Tensor<dtype>(strlen(index_char), lens, *dw);

    mttkrp_map[index][index_char] = mttkrp_map[parent_index][parent_index] * this->W[indexes[W_index]][mat_index];
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::update_cached_tensor(int left_index){
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
    while (seq_V[i]!='\0') {seq_temp[i] = seq_V[i]; i++;}
    seq_temp[left_index] = '&';
    int lens[order];
    for (int i=0; i<order; i++) {lens[i] = this->V->lens[i];}
    lens[left_index] = rank;
    Tensor<dtype> temp = Tensor<dtype>(order, lens, *this->world);
    temp[seq_temp] = (*this->V)[seq_V] * this->U[seq_U];
    cached_tensors[left_index][seq_map_init] = cached_tensors[left_index][seq_map_init] + this->VT[seq_VT]*temp[seq_temp];
    
    old_W[left_index] = this->W[left_index];
    this->is_cached[left_index] = true;
}

template<typename dtype>
void CPMSDTLROptimizer<dtype>::step() {

    World * dw = this->world;
    int order = this->order;

    // clear the Hash Table
    mttkrp_map.clear();
    // reinitialize
    update_indexes();
    // cout << left_index << endl;
    mttkrp_map_init(left_index);

    // iteration on W[i]
    for (int i=0; i<indexes.size(); i++) {
        /*  construct Matrix M
        *   M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
        */
        vector<int> mat_index = {i};

        string mat_seq;
        vec2str(mat_index, mat_seq);

        if (mttkrp_map.find(mat_seq)==mttkrp_map.end()) {
            mttkrp_map_DT(mat_seq);
        }
        Matrix<dtype> M = mttkrp_map[mat_seq];

        // calculating S
        CPOptimizer<dtype>::update_S(indexes[i]);
        // calculate gradient
        this->grad_W[indexes[i]]["ij"] = -M["ij"]+this->W[indexes[i]]["ik"]*this->S["kj"];
        if (!is_cached[indexes[i]] || (i!=(indexes.size()-1))){
            SVD_solve(M, this->W[indexes[i]], this->S);
        } else {
            get_rankR_update(this->rank, this->U, this->s, this->VT, M, this->old_W[indexes[i]], this->S);
            this->W[indexes[i]]["ij"] = this->old_W[indexes[i]]["ij"] + this->U["ik"]*this->s["k"]*this->VT["kj"];
            //update_cached_tensor(indexes[i]);
            this->low_rank_decomp = true;
        }
    }
}
