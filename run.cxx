#include "src/decomposition.h"
#include "src/CP.h"
#include "src/optimizer/cp_als_optimizer.h"
#include "src/optimizer/cp_simple_optimizer.h"
#include "src/optimizer/cp_dt_optimizer.h"
#include "src/optimizer/cp_dt_lr_optimizer.h"
#include "src/optimizer/cp_msdt_optimizer.h"
#include "src/optimizer/cp_msdt_lr_optimizer.h"

#include "common.h"
//#define ERR_REPORT

#ifndef TEST_SUITE

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char ** argv){
    int rank, np;//, n, pass;
    int const in_num = argc;
    char ** input_str = argv;

    char * model;       // 0 is CP, 1 is Tucker
    char * tensor;      // which tensor    p / p2 / c / r / r2 / o /
    int pp;             // 0 Dimention tree 1 pairwise perturbation 2 pp with <1 update_percentage_pp
    double update_percentage_pp; // pp update ratio. For each sweep only update update_percentage_pp*N matrices.
    /*
    p : poisson operator
    p2 : poisson operator with doubled dimension (decomposition is not accurate)
    c : decomposition of designed tensor with constrained collinearity
    r : decomposition of tensor made by random matrices
    r2 : random tensor
    o1 : coil-100 dataset
    */
    int dim;            // number of dimensions
    int s;              // tensor size in each dimension
    int R;              // decomposition rank
    int update_rank;    // used for optimizers with low rank updates
    int issparse;   // whether use the sparse routine or not
    double tol;     // global convergance tolerance
    double pp_res_tol;  // pp restart tolerance
    double lambda_;     // regularization param
    double magni;       // pp update magnitude
    char * filename;    // output csv filename
    double col_min;     // collinearity min
    double col_max;     // collinearity max
    double ratio_noise; // collinearity ratio of noise
    double timelimit = 5e3;  // time limits
    int maxiter = 5e3;      // maximum iterations
    int resprint = 1;
    char * tensorfile;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_File fh;

    if (getCmdOption(input_str, input_str+in_num, "-model")) {
        model = getCmdOption(input_str, input_str+in_num, "-model");
        if (model[0] != 'C' && model[0] != 'T') model = "CP";
    } else {
        model = "CP";
    }
    if (getCmdOption(input_str, input_str+in_num, "-tensor")) {
        tensor = getCmdOption(input_str, input_str+in_num, "-tensor");
    } else {
        tensor = "p";
    }
    if (getCmdOption(input_str, input_str+in_num, "-pp")) {
        pp = atoi(getCmdOption(input_str, input_str+in_num, "-pp"));
        if (pp < 0 || pp > 2) pp = 0;
    } else {
        pp = 0;
    }
    if (getCmdOption(input_str, input_str+in_num, "-update_percentage_pp")) {
        update_percentage_pp = atof(getCmdOption(input_str, input_str+in_num, "-update_percentage_pp"));
        if (update_percentage_pp < 0 || update_percentage_pp > 1) update_percentage_pp = 1.0;
    } else {
        update_percentage_pp = 1.0;
    }
    if (getCmdOption(input_str, input_str+in_num, "-dim")) {
        dim = atoi(getCmdOption(input_str, input_str+in_num, "-dim"));
        if (dim < 0) dim = 8;
    } else {
        dim = 8;
    }
    if (getCmdOption(input_str, input_str+in_num, "-maxiter")) {
        maxiter = atoi(getCmdOption(input_str, input_str+in_num, "-maxiter"));
        if (maxiter < 0) maxiter = 5e3;
    } else {
        maxiter = 5e3;
    }
    if (getCmdOption(input_str, input_str+in_num, "-timelimit")) {
        timelimit = atof(getCmdOption(input_str, input_str+in_num, "-timelimit"));
        if (timelimit < 0) timelimit = 5e3;
    } else {
        timelimit = 5e3;
    }
    if (getCmdOption(input_str, input_str+in_num, "-size")) {
        s = atoi(getCmdOption(input_str, input_str+in_num, "-size"));
        if (s < 0) s = 10;
    } else {
        s = 10;
    }
    if (getCmdOption(input_str, input_str+in_num, "-rank")) {
        R = atoi(getCmdOption(input_str, input_str+in_num, "-rank"));
        if (R < 0) R = s/2;
    } else {
        R = s/2;
    }
    if (getCmdOption(input_str, input_str+in_num, "-updaterank")) {
        update_rank = atoi(getCmdOption(input_str, input_str+in_num, "-updaterank"));
        if (update_rank < 0) update_rank = s/2;
    } else {
        update_rank = s/2;
    }
    if (getCmdOption(input_str, input_str+in_num, "-issparse")) {
        issparse = atoi(getCmdOption(input_str, input_str+in_num, "-issparse"));
        if (issparse < 0 || issparse > 1) issparse = 0;
    } else {
        issparse = 0;
    }
    if (getCmdOption(input_str, input_str+in_num, "-resprint")) {
        resprint = atoi(getCmdOption(input_str, input_str+in_num, "-resprint"));
        if (resprint < 0) resprint = 10;
    } else {
        resprint = 10;
    }
    if (getCmdOption(input_str, input_str+in_num, "-tol")) {
        tol = atof(getCmdOption(input_str, input_str+in_num, "-tol"));
        if (tol < 0 || tol > 1) tol = 1e-10;
    } else {
        tol = 1e-10;
    }
    if (getCmdOption(input_str, input_str+in_num, "-pp_res_tol")) {
        pp_res_tol = atof(getCmdOption(input_str, input_str+in_num, "-pp_res_tol"));
        if (pp_res_tol < 0 || pp_res_tol > 1) pp_res_tol = 1e-2;
    } else {
        pp_res_tol = 1e-2;
    }
    if (getCmdOption(input_str, input_str+in_num, "-lambda")) {
        lambda_ = atof(getCmdOption(input_str, input_str+in_num, "-lambda"));
        if (lambda_ < 0 ) lambda_ = 0.;
    } else {
        lambda_ = 0.;
    }
    if (getCmdOption(input_str, input_str+in_num, "-magni")) {
        magni = atof(getCmdOption(input_str, input_str+in_num, "-magni"));
        if (magni < 0 ) magni = 1.;
    } else {
        magni = 1.;
    }
    if (getCmdOption(input_str, input_str+in_num, "-filename")) {
        filename = getCmdOption(input_str, input_str+in_num, "-filename");
    } else {
        filename = "out.csv";
    }
    if (getCmdOption(input_str, input_str+in_num, "-tensorfile")) {
        tensorfile = getCmdOption(input_str, input_str+in_num, "-tensorfile");
    } else {
        tensorfile = "test";
    }
    if (getCmdOption(input_str, input_str+in_num, "-colmin")) {
        col_min = atof(getCmdOption(input_str, input_str+in_num, "-colmin"));
    } else {
        col_min = 0.5;
    }
    if (getCmdOption(input_str, input_str+in_num, "-colmax")) {
        col_max = atof(getCmdOption(input_str, input_str+in_num, "-colmax"));
    } else {
        col_max = 0.9;
    }
    if (getCmdOption(input_str, input_str+in_num, "-rationoise")) {
        ratio_noise = atof(getCmdOption(input_str, input_str+in_num, "-rationoise"));
        if (ratio_noise < 0 ) ratio_noise = 0.01;
    } else {
        ratio_noise = 0.01;
    }

    {
        double start_time = MPI_Wtime();
        World dw(argc, argv);
        srand48(dw.rank*1);

        if (dw.rank==0) {
            cout << "  model=  " << model << "  tensor=  " << tensor << "  pp=  " << pp << endl;
            cout << "  dim=  " << dim << "  size=  " << s << "  rank=  " << R << "  updaterank=  " << update_rank << endl;
            cout << "  issparse=  " << issparse << "  tolerance=  " << tol << "  restarttol=  " << pp_res_tol << endl;
            cout << "  lambda=  " << lambda_ << "  magnitude=  " << magni << "  filename=  " << filename << endl;
            cout << "  col_min=  " << col_min << "  col_max=  " << col_max  << "  rationoise  " << ratio_noise << endl;
            cout << "  timelimit=  " << timelimit << "  maxiter=  " << maxiter << "  resprint=  " << resprint  << endl;
            cout << "  tensorfile=  " << tensorfile << "  update_percentage_pp=  " << update_percentage_pp << endl;
        }

        // initialization of tensor
        Tensor<> V;

        if (tensor[0]=='p') {
            if (strlen(tensor)>1 && tensor[1]=='2') {
                //p2 : poisson operator with doubled dimension (decomposition is not accurate)
                int lens[dim];
                for (int i=0; i<dim; i++) lens[i]=s;
                V = Tensor<>(dim, issparse, lens, dw);
                laplacian_tensor(V, dim, s, issparse, dw);
            }
            else {
                //p : poisson operator
                int lens0[dim];
                for (int i=0; i<dim; i++) lens0[i]=s;
                Tensor<> V0 = Tensor<>(dim, issparse, lens0, dw);
                laplacian_tensor(V0, dim, s, issparse, dw);
                // reshape V0
                int lens[dim/2];
                for (int i=0; i<dim/2; i++) lens[i]=s*s;
                V = Tensor<>(dim/2, issparse, lens, dw);
                // reshape V0 into V
                fold_unfold(V0, V);
            }
        }
        else if (tensor[0]=='c') {
            //c : designed tensor with constrained collinearity
            int lens[dim];
            for (int i=0; i<dim; i++) lens[i]=s;
            char chars[] = {'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\0'};
            char arg[dim+1];
            arg[dim] = '\0';
            for (int i = 0; i < dim; i++) {
                arg[i] = chars[i];
            }
            V = Gen_collinearity(lens, dim, R, col_min, col_max, dw);
            Tensor<> V_noise = Tensor<>(dim, issparse, lens, dw);
            V_noise.fill_random(-1,1);
            double noise_norm = V_noise.norm2();
            double V_norm = V.norm2();
            V_noise[arg] = ratio_noise*V_norm/noise_norm*V_noise[arg];
            V[arg] = V[arg] + V_noise[arg];
        }
        else if (tensor[0]=='r') {
            if (strlen(tensor)>1 && tensor[1]=='2') {
                //r2 : random tensor
                int lens[dim];
                for (int i=0; i<dim; i++) lens[i]=s;
                V = Tensor<>(dim, issparse, lens, dw);
                V.fill_random(0.5,1);         // Why?   when V is (-1,1), low rank Tucker has no accurate decomposition
            }
            else {
                //r : tensor made by random matrices
                int lens[dim];
                for (int i=0; i<dim; i++) lens[i]=s;
                Matrix<>* W = new Matrix<>[dim];                // N matrices V will be decomposed into
                for (int i=0; i<dim; i++) {
                    W[i] = Matrix<>(s,R,dw);
                    W[i].fill_random(0,1);
                }
                build_V(V, W, dim, dw);
                delete[] W;
            }
        }
        else if (tensor[0]=='o') {
            //o1 : coil-100 dataset Rank=20 suggested
            if (strlen(tensor)>1 && tensor[1]=='1') {
                tensorfile = "coil-100.bin";
                MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &fh );
                int lens[dim];
                lens[0] = 3;
                lens[1] = 128;
                lens[2] = 128;
                lens[3] = 7200;
                // for (int i=0; i<dim; i++) lens[i]=s;
                V = Tensor<>(dim, issparse, lens, dw);
                if (dw.rank==0) cout << "Read the tensor from file coil-100 ...... " << endl;
                V.read_dense_from_file(fh);
                if (dw.rank==0) cout << "Read coil-100 dataset finished " << endl;
                // V.print();
            }
            //o2 : time-lapse dataset Rank=32 suggested
            else if (strlen(tensor)>1 && tensor[1]=='2') {
                tensorfile = "time-lapse.bin";
                MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &fh );
                int lens[dim];
                lens[0] = 33;
                lens[1] = 1344;
                lens[2] = 1024;
                lens[3] = 9;
                // for (int i=0; i<dim; i++) lens[i]=s;
                V = Tensor<>(dim, issparse, lens, dw);
                if (dw.rank==0) cout << "Read the tensor from file time-lapse ...... " << endl;
                V.read_dense_from_file(fh);
                if (dw.rank==0) cout << "Read time-lapse dataset finished " << endl;
                // V.print();
            }
        }

        double Vnorm = V.norm2();
        if (dw.rank==0) cout << "Vnorm= " << Vnorm << endl;
        ofstream Plot_File(filename);
        Matrix<>* W = new Matrix<>[V.order];                // N matrices V will be decomposed into
        Matrix<>* grad_W = new Matrix<>[V.order];           // gradients in N dimensions
        for (int i=0; i<V.order; i++) {
            Matrix<> * Wi = NULL;
            if (dw.rank == 0){
                World sworld(MPI_COMM_SELF);
                Wi = new Matrix<>(V.lens[i],R,sworld);
                Wi->fill_random(0.,1.);
            }
            W[i] = Matrix<>(V.lens[i],R,dw);
            grad_W[i] = Matrix<>(V.lens[i],R,dw);
            W[i].add_from_subworld(Wi);
            delete Wi;
        }

        // V.write_dense_to_file (fh);

        Timer_epoch tALS("ALS");
        tALS.begin();

        if (model[0]=='C') {
            if (pp==0) {
                CPD<double, CPDTOptimizer<double>> decom(dim, s, R, dw);
                decom.Init(&V, W);
                decom.als(tol*Vnorm, timelimit, maxiter, resprint, Plot_File);
                // alsCP_DT(V, W, grad_W, F, tol*Vnorm, timelimit, maxiter, lambda_, Plot_File, resprint, false, dw);
            }
            else if (pp==1) {
                CPD<double, CPMSDTOptimizer<double>> decom(dim, s, R, dw);
                decom.Init(&V, W);
                decom.als(tol*Vnorm, timelimit, maxiter, resprint, Plot_File);
            //  alsCP_PP(V, W, grad_W, F, tol*Vnorm, pp_res_tol, timelimit, maxiter, lambda_, magni, Plot_File, resprint, false, dw);
            }
            else if (pp==2){
                CPD<double, CPDTLROptimizer<double>> decom(dim, s, R, update_rank, dw);
                decom.Init(&V, W);
                decom.als(tol*Vnorm, timelimit, maxiter, resprint, Plot_File);
            }
            else if (pp==3){
                CPD<double, CPMSDTLROptimizer<double>> decom(dim, s, R, update_rank, dw);
                decom.Init(&V, W);
                decom.als(tol*Vnorm, timelimit, maxiter, resprint, Plot_File);
            }
            // else if (pp==2) {
            //  alsCP_PP_partupdate(V, W, grad_W, F, tol*Vnorm, pp_res_tol, timelimit, maxiter, lambda_, magni, update_percentage_pp, Plot_File, resprint, false, dw);
            // }
        }
        // else if (model[0]=='T') {
        //  int ranks[V.order];
        //  if (tensor[0]=='o') {
        //      //o1 : coil-100 dataset
        //      if (strlen(tensor)>1 && tensor[1]=='1') {
        //          ranks[0] = 3;
        //          ranks[1] = 10;
        //          ranks[2] = 10;
        //          ranks[3] = 70;
        //      }
        //      //o2 : time-lapse dataset
        //      else if (strlen(tensor)>1 && tensor[1]=='2') {
        //          ranks[0] = 10;
        //          ranks[1] = 100;
        //          ranks[2] = 100;
        //          ranks[3] = 5;
        //      }
        //  } else {
        //      for (int i=0; i<V.order; i++) {
        //          ranks[i] = R;
        //      }
        //  }
        //  Tensor<> hosvd_core;
        //  // using hosvd to initialize W and hosvd_core
        //  hosvd(V, hosvd_core, W, ranks, dw);
        //  if (pp==0) {
        //      alsTucker_DT(V, hosvd_core, W, tol*Vnorm, timelimit, maxiter, Plot_File, resprint, false, dw);
        //  }
        //  else if (pp==1) {
        //      alsTucker_PP(V, hosvd_core, W, tol*Vnorm, pp_res_tol, timelimit, maxiter, Plot_File, resprint, false, dw);
        //  }
        // }

        tALS.end();
        if(dw.rank==0) {
            printf ("experiment took %lf seconds\n",MPI_Wtime()-start_time);
        }

        // delete[] W;
        // delete[] grad_W;
        // delete[] F;

        if (tensor[0]=='o') {
            MPI_File_close( &fh );
        }

    }

    MPI_Finalize();
    return 0;
}

#endif
