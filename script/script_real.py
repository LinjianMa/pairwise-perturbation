import os


jobname = "ctf_pp_real"     
queue = "normal"        
nodes = 16        
mpitask = 16                
time = "10:00:00"           
mail = "solomon2@illinois.edu"           
mailtype = "all"

CTF_PPN = 8
OMP_NUM_THREADS = 6
executive = './test_ALS'

# myproject = "project"      #SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

try:
    os.stat('real_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))
except:
    os.mkdir('real_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))    
os.chdir('real_data_nodes='+str(nodes)+'PPN='+str(CTF_PPN)+'THREADS='+str(OMP_NUM_THREADS))

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

text_file.write("ibrun %s -model CP -tensor o1 -pp 0 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=%s_pp=0_rank=10.csv -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model CP -tensor o1 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=%s_pp=1_rank=10_restol=0.05.csv -restol 0.05 -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model CP -tensor o1 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o1_nodes=%s_pp=1_rank=10_restol=0.1.csv -restol 0.1 -resprint 10\n\n" % (executive,nodes))

text_file.write("ibrun %s -model CP -tensor o2 -pp 0 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=%s_pp=0_rank=10.csv -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model CP -tensor o2 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=%s_pp=1_rank=10_restol=0.05.csv -restol 0.05 -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model CP -tensor o2 -pp 1 -dim 4 -rank 10 -maxiter 250 -filename CP_o2_nodes=%s_pp=1_rank=10_restol=0.1.csv -restol 0.1 -resprint 10\n\n" % (executive,nodes))

text_file.write("ibrun %s -model Tucker -tensor o1 -pp 0 -dim 4 -maxiter 250 -filename Tucker_o1_nodes=%s_pp=0.csv -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model Tucker -tensor o1 -pp 1 -dim 4 -maxiter 250 -filename Tucker_o1_nodes=%s_pp=1_restol=0.5.csv -restol 0.5 -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model Tucker -tensor o1 -pp 1 -dim 4 -maxiter 250 -filename Tucker_o1_nodes=%s_pp=1_restol=0.1.csv -restol 0.1 -resprint 10\n\n" % (executive,nodes))

text_file.write("ibrun %s -model Tucker -tensor o2 -pp 0 -dim 4 -maxiter 250 -filename Tucker_o2_nodes=%s_pp=0.csv -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model Tucker -tensor o2 -pp 1 -dim 4 -maxiter 250 -filename Tucker_o2_nodes=%s_pp=1_restol=0.5.csv -restol 0.5 -resprint 10\n" % (executive,nodes))
text_file.write("ibrun %s -model Tucker -tensor o2 -pp 1 -dim 4 -maxiter 250 -filename Tucker_o2_nodes=%s_pp=1_restol=0.1.csv -restol 0.1 -resprint 10\n\n" % (executive,nodes))


text_file.close()






