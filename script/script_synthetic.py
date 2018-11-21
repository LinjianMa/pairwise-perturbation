import os


jobname = "ctf_pp_synthetic"     
# outfile = "pp.out"      
# errorfile = "pp.error"    
queue = "normal"        
nodes = 1        
mpitask = 16                
time = "10:00:00"           
mail = "solomon2@illinois.edu"           
mailtype = "all"

CTF_PPN = 8
OMP_NUM_THREADS = 6
executive = './test_ALS'

# myproject = "project"      #SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

try:
    os.stat('synthetic_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))
except:
    os.mkdir('synthetic_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))    
os.chdir('synthetic_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))

text_file = open("run.sh", "w")

text_file.write("#!/bin/bash\n")
text_file.write("#----------------------------------------------------\n\n\n")
text_file.write("#SBATCH -J %s\n" % jobname)
text_file.write("#SBATCH -o pp.bench.N%s.n%s.o%%j.out\n" % (nodes,mpitask))
text_file.write("#SBATCH -e pp.bench.N%s.n%s.o%%j.err\n" % (nodes,mpitask))
text_file.write("#SBATCH -p %s\n" % queue)
text_file.write("#SBATCH -N %s\n" % nodes)
text_file.write("#SBATCH -n %s\n" % mpitask)
text_file.write("#SBATCH -t %s\n" % time)
text_file.write("#SBATCH --mail-user= %s\n" % mail)
text_file.write("#SBATCH --mail-type= %s\n\n" % mailtype)
# text_file.write("#SBATCH -A %s\n\n\n" % myproject)

text_file.write("module list\n")
text_file.write("pwd\n")
text_file.write("date\n\n")

text_file.write("export CTF_PPN=%s\n" % CTF_PPN)
text_file.write("export OMP_NUM_THREADS=%s\n\n" % OMP_NUM_THREADS)

dim = 6
size = int(30*nodes**(1./dim))

text_file.write("ibrun %s -model CP -tensor r -pp 0 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_r_nodes=%s_pp=0_dim=6_size=%s_rank=5.csv -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor r -pp 1 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_r_nodes=%s_pp=1_dim=6_size=%s_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor r -pp 1 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_r_nodes=%s_pp=1_dim=6_size=%s_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10\n\n" % (executive,size,nodes,size))

text_file.write("ibrun %s -model CP -tensor c -pp 0 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_c_nodes=%s_pp=0_dim=6_size=%s_rank=5.csv -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor c -pp 1 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_c_nodes=%s_pp=1_dim=6_size=%s_rank=5_restol=0.01.csv -pp_res_tol 0.01 -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor c -pp 1 -dim 6 -size %s -rank 5 -maxiter 250 -filename CP_c_nodes=%s_pp=1_dim=6_size=%s_rank=5_restol=0.05.csv -pp_res_tol 0.05 -resprint 10\n\n" % (executive,size,nodes,size))

dim = 8
size = int(13*nodes**(1./dim))

text_file.write("ibrun %s -model CP -tensor p -pp 0 -dim 6 -size %s -rank 2 -maxiter 250 -filename CP_p_nodes=%s_pp=0_dim=8_size=%s_rank=2.csv -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor p -pp 1 -dim 6 -size %s -rank 2 -maxiter 250 -filename CP_p_nodes=%s_pp=1_dim=8_size=%s_rank=2_restol=0.01.csv -pp_res_tol 0.01 -resprint 10\n" % (executive,size,nodes,size))
text_file.write("ibrun %s -model CP -tensor p -pp 1 -dim 6 -size %s -rank 2 -maxiter 250 -filename CP_p_nodes=%s_pp=1_dim=8_size=%s_rank=2_restol=0.05.csv -pp_res_tol 0.05 -resprint 10\n\n" % (executive,size,nodes,size))


text_file.close()
