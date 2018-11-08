#!/bin/bash
#----------------------------------------------------
#SBATCH -J ctf_pp
#SBATCH -o pp.bench.N1.n8.1e-2.o%j.out
#SBATCH -e pp.bench.N1.n8.1e-2.o%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH --mail-user=solomon2@illinois.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

module list
pwd
date

export CTF_PPN=8
export OMP_NUM_THREADS=6

ibrun ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 13 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=0_dim=8_size=13_rank=2.csv -resprint 10
ibrun ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 13 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=8_size=13_rank=2_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 13 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=8_size=13_rank=2_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10


ibrun ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_p2_nodes=1_pp=0_dim=6_size=22_rank=6.csv -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_p2_nodes=1_pp=1_dim=6_size=22_rank=6_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_p2_nodes=1_pp=1_dim=6_size=22_rank=6_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10


ibrun ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_r2_nodes=1_pp=0_dim=6_size=22_rank=6.csv -resprint 10
ibrun ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_r2_nodes=1_pp=1_dim=6_size=22_rank=6_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 22 -rank 6 -maxiter 250 -filename Tucker_r2_nodes=1_pp=1_dim=6_size=22_rank=6_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10






