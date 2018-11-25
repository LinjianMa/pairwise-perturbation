import os

jobname = "ctf_pp_weakscaling"
# outfile = "pp.out"
# errorfile = "pp.error"
CTF_PPN = 8
OMP_NUM_THREADS = 8
queue = "normal"
for nodes in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    mpitask = nodes * CTF_PPN
    time = "01:00:00"
    mail = "solomon2@illinois.edu"
    mailtype = "all"

    exe = './pp_bench'

    # myproject = "project"      #SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

    text_file = open("script_ws_%s.sh" % nodes, "w")

    text_file.write("#!/bin/bash\n")
    text_file.write("#----------------------------------------------------\n\n\n")
    text_file.write("#SBATCH -J %s\n" % jobname)
    text_file.write("#SBATCH -o pp.bench.ws.N%s.n%s.o%%j.out\n" % (nodes,mpitask))
    text_file.write("#SBATCH -e pp.bench.ws.N%s.n%s.o%%j.err\n" % (nodes,mpitask))
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

    size = int(32*nodes**(1./6))
    rank = int(4*nodes**(1./6))

    text_file.write("ibrun test_suite\n")
    text_file.write("ibrun %s -model CP -tensor r -dim 6 -size %s -rank %s -maxiter 5 -filename CP_r_nodes=%s_processes=%s_dim=6_size=%s_rank=%s_bench.csv -resprint 1\n" % (exe,size,rank,nodes,mpitask,size,rank))
    text_file.write("ibrun %s -model Tucker -tensor r2 -dim 6 -size %s -rank %s -maxiter 5 -filename Tucker_r2_nodes=%s_processes=%s_dim=6_size=%s_rank=%s_bench.csv -resprint 1\n" % (exe,size,rank,nodes,mpitask,size,rank))


    text_file.close()
