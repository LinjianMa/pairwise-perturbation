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


mpirun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 5 -rank 2 -maxiter 2500 -filename CP_p_nodes=1_pp=0_dim=8_size=5_rank=2.csv
mpirun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 5 -rank 2 -maxiter 2500 -filename CP_p_nodes=1_pp=1_dim=8_size=5_rank=2.csv
mpirun -np 1 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 4 -rank 2 -maxiter 2500 -filename CP_p_nodes=1_pp=0_dim=12_size=4_rank=2.csv
mpirun -np 1 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 4 -rank 2 -maxiter 2500 -filename CP_p_nodes=1_pp=1_dim=12_size=4_rank=2.csv
mpirun -np 1 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 10 -rank 8 -maxiter 2500 -filename CP_p2_nodes=1_pp=0_dim=6_size=10_rank=8.csv
mpirun -np 1 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 10 -rank 8 -maxiter 2500 -filename CP_p2_nodes=1_pp=1_dim=6_size=10_rank=8.csv
mpirun -np 1 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 14 -rank 5 -maxiter 2500 -filename CP_r_nodes=1_pp=0_dim=6_size=14_rank=5.csv
mpirun -np 1 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 14 -rank 5 -maxiter 2500 -filename CP_r_nodes=1_pp=1_dim=6_size=14_rank=5.csv
mpirun -np 1 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 14 -rank 5 -maxiter 2500 -filename CP_c_nodes=1_pp=0_dim=6_size=14_rank=5.csv
mpirun -np 1 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 14 -rank 5 -maxiter 2500 -filename CP_c_nodes=1_pp=1_dim=6_size=14_rank=5.csv


mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 4 -size 40 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=0_dim=4_size=40_rank=10.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 4 -size 40 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=1_dim=4_size=40_rank=10.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 16 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=0_dim=6_size=16_rank=8.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 16 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=1_dim=6_size=16_rank=8.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 16 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=0_dim=6_size=16_rank=5.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 16 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=1_pp=1_dim=6_size=16_rank=5.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 4 -size 40 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=0_dim=4_size=40_rank=10.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 4 -size 40 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=1_dim=4_size=40_rank=10.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 16 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=0_dim=6_size=16_rank=5.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 16 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=1_dim=6_size=16_rank=5.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 16 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=0_dim=6_size=16_rank=8.csv
mpirun -np 1 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 16 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=1_pp=1_dim=6_size=16_rank=8.csv


mpirun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 7 -rank 2 -maxiter 2500 -filename CP_p_nodes=16_pp=0_dim=8_size=7_rank=2.csv
mpirun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 7 -rank 2 -maxiter 2500 -filename CP_p_nodes=16_pp=1_dim=8_size=7_rank=2.csv
mpirun -np 16 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 5 -rank 2 -maxiter 2500 -filename CP_p_nodes=16_pp=0_dim=12_size=5_rank=2.csv
mpirun -np 16 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 5 -rank 2 -maxiter 2500 -filename CP_p_nodes=16_pp=1_dim=12_size=5_rank=2.csv
mpirun -np 16 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 15 -rank 8 -maxiter 2500 -filename CP_p2_nodes=16_pp=0_dim=6_size=15_rank=8.csv
mpirun -np 16 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 15 -rank 8 -maxiter 2500 -filename CP_p2_nodes=16_pp=1_dim=6_size=15_rank=8.csv
mpirun -np 16 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 22 -rank 5 -maxiter 2500 -filename CP_r_nodes=16_pp=0_dim=6_size=22_rank=5.csv
mpirun -np 16 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 22 -rank 5 -maxiter 2500 -filename CP_r_nodes=16_pp=1_dim=6_size=22_rank=5.csv
mpirun -np 16 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 22 -rank 5 -maxiter 2500 -filename CP_c_nodes=16_pp=0_dim=6_size=22_rank=5.csv
mpirun -np 16 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 22 -rank 5 -maxiter 2500 -filename CP_c_nodes=16_pp=1_dim=6_size=22_rank=5.csv


mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 4 -size 80 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=0_dim=4_size=80_rank=10.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 4 -size 80 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=1_dim=4_size=80_rank=10.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=0_dim=6_size=25_rank=8.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=25_rank=8.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 25 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=0_dim=6_size=25_rank=5.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 25 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=16_pp=1_dim=6_size=25_rank=5.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 4 -size 80 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=0_dim=4_size=80_rank=10.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 4 -size 80 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=1_dim=4_size=80_rank=10.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 25 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=0_dim=6_size=25_rank=5.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 25 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=1_dim=6_size=25_rank=5.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=0_dim=6_size=25_rank=8.csv
mpirun -np 16 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=16_pp=1_dim=6_size=25_rank=8.csv


mpirun -np 256 ./test_ALS -model CP -tensor p -pp 0 -dim 8 -size 10 -rank 2 -maxiter 2500 -filename CP_p_nodes=256_pp=0_dim=8_size=10_rank=2.csv
mpirun -np 256 ./test_ALS -model CP -tensor p -pp 1 -dim 8 -size 10 -rank 2 -maxiter 2500 -filename CP_p_nodes=256_pp=1_dim=8_size=10_rank=2.csv
mpirun -np 256 ./test_ALS -model CP -tensor p -pp 0 -dim 12 -size 6 -rank 2 -maxiter 2500 -filename CP_p_nodes=256_pp=0_dim=12_size=6_rank=2.csv
mpirun -np 256 ./test_ALS -model CP -tensor p -pp 1 -dim 12 -size 6 -rank 2 -maxiter 2500 -filename CP_p_nodes=256_pp=1_dim=12_size=6_rank=2.csv
mpirun -np 256 ./test_ALS -model CP -tensor p2 -pp 0 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename CP_p2_nodes=256_pp=0_dim=6_size=25_rank=8.csv
mpirun -np 256 ./test_ALS -model CP -tensor p2 -pp 1 -dim 6 -size 25 -rank 8 -maxiter 2500 -filename CP_p2_nodes=256_pp=1_dim=6_size=25_rank=8.csv
mpirun -np 256 ./test_ALS -model CP -tensor r -pp 0 -dim 6 -size 35 -rank 5 -maxiter 2500 -filename CP_r_nodes=256_pp=0_dim=6_size=35_rank=5.csv
mpirun -np 256 ./test_ALS -model CP -tensor r -pp 1 -dim 6 -size 35 -rank 5 -maxiter 2500 -filename CP_r_nodes=256_pp=1_dim=6_size=35_rank=5.csv
mpirun -np 256 ./test_ALS -model CP -tensor c -pp 0 -dim 6 -size 35 -rank 5 -maxiter 2500 -filename CP_c_nodes=256_pp=0_dim=6_size=35_rank=5.csv
mpirun -np 256 ./test_ALS -model CP -tensor c -pp 1 -dim 6 -size 35 -rank 5 -maxiter 2500 -filename CP_c_nodes=256_pp=1_dim=6_size=35_rank=5.csv


mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 4 -size 160 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=0_dim=4_size=160_rank=10.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 4 -size 160 -rank 10 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=1_dim=4_size=160_rank=10.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 40 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=0_dim=6_size=40_rank=8.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 40 -rank 8 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=1_dim=6_size=40_rank=8.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 0 -dim 6 -size 40 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=0_dim=6_size=40_rank=5.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor p2 -pp 1 -dim 6 -size 40 -rank 5 -maxiter 2500 -filename Tucker_p2_nodes=256_pp=1_dim=6_size=40_rank=5.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 4 -size 160 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=0_dim=4_size=160_rank=10.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 4 -size 160 -rank 10 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=1_dim=4_size=160_rank=10.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 40 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=0_dim=6_size=40_rank=5.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 40 -rank 5 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=1_dim=6_size=40_rank=5.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 0 -dim 6 -size 40 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=0_dim=6_size=40_rank=8.csv
mpirun -np 256 ./test_ALS -model Tucker -tensor r2 -pp 1 -dim 6 -size 40 -rank 8 -maxiter 2500 -filename Tucker_r2_nodes=256_pp=1_dim=6_size=40_rank=8.csv


