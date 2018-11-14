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

ibrun ./test_ALS -model CP -tensor o1 -pp 0 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=16_pp=0_rank=10.csv -resprint 10
ibrun ./test_ALS -model CP -tensor o1 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=16_pp=1_rank=10_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model CP -tensor o1 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=16_pp=1_rank=10_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10

ibrun ./test_ALS -model CP -tensor o2 -pp 0 -dim 4 -rank 5 -maxiter 250 -filename CP_o2_nodes=16_pp=0_rank=5.csv -resprint 10
ibrun ./test_ALS -model CP -tensor o2 -pp 1 -dim 4 -rank 5 -maxiter 250 -filename CP_o2_nodes=16_pp=1_rank=5_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model CP -tensor o2 -pp 1 -dim 4 -rank 5 -maxiter 250 -filename CP_o2_nodes=16_pp=1_rank=5_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10

ibrun ./test_ALS -model CP -tensor o2 -pp 0 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=16_pp=0_rank=10.csv -resprint 10
ibrun ./test_ALS -model CP -tensor o2 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=16_pp=1_rank=10_restol=1e-2.csv -pp_res_tol 1e-2 -resprint 10
ibrun ./test_ALS -model CP -tensor o2 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=16_pp=1_rank=10_restol=5e-2.csv -pp_res_tol 5e-2 -resprint 10




