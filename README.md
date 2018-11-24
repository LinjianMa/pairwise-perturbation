## Pairwise Perturbation for Tensor Decomposition

Pairwise Perturbation (PP) is an efficient numerical algorithm for alternating least squares (ALS) in CP and Tucker decompositions.

PP uses perturbative corrections to the ALS subproblems rather than recomputing the tensor contractions.

This repository implements PP in C++, and is based on the parallel numerical library [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf). 

## Building

To correctly run the code, CTF should be installed at first. See [CTF](https://github.com/cyclops-community/ctf) repository for detailed instructions. 

Modify `config.mk` based on your local information.  

## Run coil-100 real data test: 

Use the scripts in the script folder as follows to download the coil-100 dataset and change the data into binary file:
```
./get_coil.sh
python imageloader.py
```
Then run `test_ALS` with following command:
```
mpirun n -model CP -tensor o1 -pp 0 -dim 4 -rank 10 -maxiter 250
```

## Run time-lapse real data test: 
Use the scripts in the script folder as follows to download the time-lapse dataset and change the data into binary file:
```
./get_time_lapse.sh
./unzip_time_lapse.sh
python matloader.py
```
Then run `test_ALS` with following command:
```
mpirun n -model CP -tensor o2 -pp 0 -dim 4 -rank 10 -maxiter 250
```

## Benchmark experiments and tests on synthetic data

Commands for these tests are in `script_strongscaling.py`, `script_weakscaling.py`, `script_synthetic.py`, `script_real.py` in the script folder. 