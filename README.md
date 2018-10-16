## collsec

Repository for the code of the paper titled "On Collaborative Predictive Blacklisting" by Luca Melis, Apostolos Pyrgelis 
and Emiliano De Cristofaro (UCL). A preliminary version of the paper can be found on the following link:

https://arxiv.org/abs/1810.02649

## Setup

Our experiments have been executed on a desktop PC with an Intel(R) Core(TM) i7-4790 CPU and 16GB of RAM 
running Ubuntu 16.04 LTS. To repeat the experiments, install python 2.7 as well as the following Python packages: 

- numpy 1.14.0 
- scipy 1.0.0 
- scikit-learn 0.19.1 
- pandas 0.22.0 
- matplotlib 2.0.0 

All the above packages can be installed via the tool ’pip’.

## Dataset

To obtain the DShield dataset that was used in our experiments, use the following download link 
and extract its contents (i.e., the '.pkl' files) in the ’data’ folder of the repository:

https://www.dropbox.com/s/kmiejttl4ceufpp/data.zip

## Implicit Recommendation by Soldo et al.

https://ieeexplore.ieee.org/abstract/document/5461982/

To replicate the experiments for Soldo et al’s recommendation system we rely on the MATLAB 
implementation of Chakrabarti et al. (https://dl.acm.org/citation.cfm?id=1014064)
for the Cross Associations (CA) co-clustering algorithm. To this end, one should install:

- Octave 4.0.0 
- oct2py 3.5.0 

To compile the Cross Associations algorithm follow the steps:

>> cd soldo/CA_python

>> octave

>> mex cc_col_nz.c

>> quit

To link our python implementation with the Octave workspace of CA configure accordingly the path 
in the file ’soldo/CA_python/ca_utils.py’ (line 6). To run the experiments:

>> cd soldo

>> python soldo.py

To configure the parameter k of the k-NN algorithm included in the ensemble method modify the file 
’soldo/top_neighbors.py’. If experiments for various values of k are executed, modify the file 'soldo/soldo.py' 
to prevent the CA algorithm from running again.

## Controlled Data Sharing by Freudiger et al.

https://link.springer.com/chapter/10.1007%2F978-3-319-20550-2_17

To repeat the experiments for the Controlled Data Sharing system by Freudiger et al., perform the following steps:

### Approach A

>> cd dimva-global

>> python dimva-global.py

### Approach B

>> cd dimva−local

>> python dimva−local.py

Note. To configure the length of the training and testing windows modify the file ’utils/dimva_util.py’.

## Hybrid Scheme

To replicate the experiments for our proposed hybrid scheme simply execute the following steps:

### Agglomerative Clustering:

>> cd agglomerative

>> python agglomerative.py

### K-Means Clustering:

>> cd kmeans

>> python kmeans.py

### Nearest Neighbors:

>> cd knn

>> python knn.py

By default is configured we utilize a 5-day training window and a 1-day testing one 
as done in previous work. To modify this setting, adjust the parameters 
indicated in the files ’utils/util.py’ and ’utils/time_series.py’.

## Results

The results of all the above scripts are stored in the folder titled ’results’. To visualize the results 
with matplotlib and obtain the figures presented in the paper, execute:

>> cd collsec/results

>> python hybrid_plots.py

>> python dimva_global_plots.py

>> python dimva_local_plots.py

>> python soldo_plots.py
