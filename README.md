# pairwise-perturbation

## To run coil-100 real data test: 
1. Download dataset from  http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php

2. Change the path to the dataset in imageloader.py

3. Run 
```
python imageloader.py
```

4. Run 
```
mpirun -np 8 ./test_ALS  -dim 4 -rank 10 -tensor o1      

```
