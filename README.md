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


## To run time-lapse real data test: 
1. Download dataset from  https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/Time-Lapse_HSIs_2015.html

Download 9 datasets under "Nogueiró scene (acquired 9 June 2003)"

Unzip all of them. 

2. Change the path to the dataset in matloader.py

3. Run 
```
python matloader.py
```

4. Run 
```
mpirun -np 8 ./test_ALS  -dim 4 -rank 5 -tensor o2      

```