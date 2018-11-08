# import os

text_file = open("run.sh", "w")
#!/bin/bash
#----------------------------------------------------
jobname = "ppurturbation"      #SBATCH -J myjob           # Job name
outfile = "pp.out"      #SBATCH -o myjob.o%j       # Name of stdout output file
errorfile = "pp.error"    #SBATCH -e myjob.e%j       # Name of stderr error file
queue = "normal"        #SBATCH -p normal          # Queue (partition) name
nodes = 256          #SBATCH -N 1               # Total # of nodes (must be 1 for serial)
mpitask = 1                #SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
time = "10:00:00"           #SBATCH -t 01:30:00        # Run time (hh:mm:ss)
mail = "xxx@illinois.edu"           #SBATCH --mail-user=myname@myschool.edu
mailtype = "all"          #SBATCH --mail-type=all    # Send email at begin and end of job
myproject = "project"      #SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

text_file.write("#!/bin/bash\n")
text_file.write("#----------------------------------------------------\n\n\n")
text_file.write("#SBATCH -J %s\n" % jobname)
text_file.write("#SBATCH -o %s\n" % outfile)
text_file.write("#SBATCH -e %s\n" % errorfile)
text_file.write("#SBATCH -p %s\n" % queue)
text_file.write("#SBATCH -N %s\n" % nodes)
text_file.write("#SBATCH -n %s\n" % mpitask)
text_file.write("#SBATCH -t %s\n" % time)
text_file.write("#SBATCH --mail-user= %s\n" % mail)
text_file.write("#SBATCH --mail-type= %s\n" % mailtype)
text_file.write("#SBATCH -A %s\n\n\n" % myproject)


mpirun = "ibrun"
nodes = [1,16]
exefile = "./test_ALS"
maxiter = 250
pp_res_tols = [1e-2, 5e-2]


for node in nodes: 
	for pp_res_tol in pp_res_tols:

		# CP test examples
		# 1: TEST_3d_poisson_CP(8, 5, 2, 0, 1e-10, 1e-3, 0.00, 1, Plot_File, dw); 
		model = "CP"
		tensor = "p"
		pp = 0
		dim = 8
		size = int(15*node**(1./dim))
		rank = 2
		# pp_res_tol = "1e-2"
		resprint = 10

		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		pp = 1
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		#2: TEST_3d_poisson_CP(12, 4, 2, 0, 1e-10, 1e-3, 0.00, 1.0, Plot_File, dw);
		model = "CP"
		tensor = "p"
		pp = 0
		dim = 12
		size = int(6*node**(1./dim))
		rank = 2
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		pp = 1
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		#3: TEST_poisson_CP(6, 10, 8, 0, 1e-3, 0.00, 0.8, Plot_File, dw);
		model = "CP"
		tensor = "p2"
		pp = 0
		dim = 6
		size = int(30*node**(1./dim))
		rank = 8
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		pp = 1
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


		#4: TEST_randmat_CP(6, 14, 5, false, 1e-10, 1e-3, 0.00, 1., Plot_File, dw);
		model = "CP"
		tensor = "r"
		pp = 0
		dim = 6
		size = int(30*node**(1./dim))
		rank = 5
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		pp = 1
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


		#5: TEST_collinearity_CP(6, 14,	5, false, 1e-10, 1e-3, 0.00, 1., 0.5, 0.9, 0.05, Plot_File, dw);
		model = "CP"
		tensor = "c"
		pp = 0
		dim = 6
		size = int(30*node**(1./dim))
		rank = 5
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		pp = 1
		filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+"_restol="+str(pp_res_tol)+".csv"
		text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
			% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

		text_file.write("\n\n")

	# #Tucker test examples:
	# #1. TEST_sparse_laplacian_alsTucker(4, 40, 10, 0, 1e-10, Plot_File, dw); 
	# model = "Tucker"
	# tensor = "p2"
	# pp = 0
	# dim = 4
	# size = int(100*node**(1./dim))
	# rank = 10
	# pp_res_tol = "5e-2"
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


	# #2. TEST_sparse_laplacian_alsTucker(6, 16, 8, 0, 1e-10, Plot_File, dw); 
	# model = "Tucker"
	# tensor = "p2"
	# pp = 0
	# dim = 6
	# size = int(16*node**(1./dim))
	# rank = 8
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


	# #3. TEST_sparse_laplacian_alsTucker(6, 16, 5, 0, 1e-10, Plot_File, dw); 
	# model = "Tucker"
	# tensor = "p2"
	# pp = 0
	# dim = 6
	# size = int(16*node**(1./dim))
	# rank = 5
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


	# #4. TEST_random_alsTucker(4, 40, 10, 0, 1e-10, Plot_File, dw);
	# model = "Tucker"
	# tensor = "r2"
	# pp = 0
	# dim = 4
	# size = int(100*node**(1./dim))
	# rank = 10
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))



	# #5. TEST_random_alsTucker(6, 16, 5, 0, 1e-10, Plot_File, dw); 
	# model = "Tucker"
	# tensor = "r2"
	# pp = 0
	# dim = 6
	# size = int(16*node**(1./dim))
	# rank = 5
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))


	# #6. TEST_random_alsTucker(6, 16, 8, 0, 1e-10, Plot_File, dw); 
	# model = "Tucker"
	# tensor = "r2"
	# pp = 0
	# dim = 6
	# size = int(16*node**(1./dim))
	# rank = 8
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	# pp = 1
	# filename = model+"_"+tensor+"_"+"nodes="+str(node)+"_pp="+str(pp)+"_dim="+str(dim)+"_size="+str(size)+"_rank="+str(rank)+".csv"
	# text_file.write("%s -np %s %s -model %s -tensor %s -pp %s -dim %s -size %s -rank %s -maxiter %s -filename %s -pp_res_tol %s -resprint %s\n" \
	# 	% (mpirun, node, exefile, model, tensor, pp, dim, size, rank, maxiter, filename, pp_res_tol, resprint))

	text_file.write("\n\n")

text_file.close()
