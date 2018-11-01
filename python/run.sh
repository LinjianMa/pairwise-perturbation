#!/bin/bash
#----------------------------------------------------


#SBATCH -J ppurturbation
#SBATCH -o pp.out
#SBATCH -e pp.error
#SBATCH -p normal
#SBATCH -N 256
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mail-user= xxx@illinois.edu
#SBATCH --mail-type= all
#SBATCH -A project


ibrun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 15 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=0_dim=8_size=15_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 15 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=8_size=15_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 6 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=0_dim=12_size=6_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 6 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=12_size=6_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 30 -rank 8 -maxiter 250 -filename CP_p2_nodes=1_pp=0_dim=6_size=30_rank=8_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 30 -rank 8 -maxiter 250 -filename CP_p2_nodes=1_pp=1_dim=6_size=30_rank=8_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_r_nodes=1_pp=0_dim=6_size=30_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_r_nodes=1_pp=1_dim=6_size=30_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_c_nodes=1_pp=0_dim=6_size=30_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_c_nodes=1_pp=1_dim=6_size=30_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10


ibrun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 15 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=0_dim=8_size=15_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 15 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=8_size=15_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 6 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=0_dim=12_size=6_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 6 -rank 2 -maxiter 250 -filename CP_p_nodes=1_pp=1_dim=12_size=6_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 30 -rank 8 -maxiter 250 -filename CP_p2_nodes=1_pp=0_dim=6_size=30_rank=8_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 30 -rank 8 -maxiter 250 -filename CP_p2_nodes=1_pp=1_dim=6_size=30_rank=8_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_r_nodes=1_pp=0_dim=6_size=30_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_r_nodes=1_pp=1_dim=6_size=30_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_c_nodes=1_pp=0_dim=6_size=30_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 1 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 30 -rank 5 -maxiter 250 -filename CP_c_nodes=1_pp=1_dim=6_size=30_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10




ibrun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 21 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=0_dim=8_size=21_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 21 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=8_size=21_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 7 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=0_dim=12_size=7_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 7 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=12_size=7_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 47 -rank 8 -maxiter 250 -filename CP_p2_nodes=16_pp=0_dim=6_size=47_rank=8_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 47 -rank 8 -maxiter 250 -filename CP_p2_nodes=16_pp=1_dim=6_size=47_rank=8_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_r_nodes=16_pp=0_dim=6_size=47_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_r_nodes=16_pp=1_dim=6_size=47_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_c_nodes=16_pp=0_dim=6_size=47_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_c_nodes=16_pp=1_dim=6_size=47_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10


ibrun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 21 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=0_dim=8_size=21_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 21 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=8_size=21_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 7 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=0_dim=12_size=7_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 7 -rank 2 -maxiter 250 -filename CP_p_nodes=16_pp=1_dim=12_size=7_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 47 -rank 8 -maxiter 250 -filename CP_p2_nodes=16_pp=0_dim=6_size=47_rank=8_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 47 -rank 8 -maxiter 250 -filename CP_p2_nodes=16_pp=1_dim=6_size=47_rank=8_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_r_nodes=16_pp=0_dim=6_size=47_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_r_nodes=16_pp=1_dim=6_size=47_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_c_nodes=16_pp=0_dim=6_size=47_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10
ibrun -np 16 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 47 -rank 5 -maxiter 250 -filename CP_c_nodes=16_pp=1_dim=6_size=47_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10




