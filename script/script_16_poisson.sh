#!/bin/bash
#----------------------------------------------------
#SBATCH -J ctf_pp
#SBATCH -o pp.bench.N16.n128.1e-2.o%j.out
#SBATCH -e pp.bench.N16.n128.1e-2.o%j.err
#SBATCH -p normal
#SBATCH -N 16
#SBATCH -n 128
#SBATCH -t 04:00:00
#SBATCH --mail-user=solomon2@illinois.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

module list
pwd
date

export CTF_PPN=8
export OMP_NUM_THREADS=6

# ibrun ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 18 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=0_dim=8_size=18_rank=2.csv -resprint 10
# ibrun ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 18 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=8_size=18_rank=2_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
# ibrun ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 18 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=8_size=18_rank=2_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10


ibrun ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 33 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=0_dim=6_size=33_rank=9.csv -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 33 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=33_rank=9_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 33 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=33_rank=9_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10

ibrun ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 32 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=0_dim=6_size=32_rank=9.csv -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 32 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=32_rank=9_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 32 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=32_rank=9_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10

ibrun ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 30 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=0_dim=6_size=30_rank=9.csv -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 30 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=30_rank=9_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 30 -rank 9 -maxiter 250 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=30_rank=9_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10


# ibrun ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 34 -rank 10 -maxiter 250 -filename Tucker_r2_nodes=16_pp=0_dim=6_size=34_rank=10.csv -resprint 10
# ibrun ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 34 -rank 10 -maxiter 250 -filename Tucker_r2_nodes=16_pp=1_dim=6_size=34_rank=10_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
# ibrun ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 34 -rank 10 -maxiter 250 -filename Tucker_r2_nodes=16 _pp=1_dim=6_size=34_rank=10_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10


