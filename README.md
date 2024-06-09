**SMDE**

This is the code corresponding to the experiments conducted for the work "SMDE:Unsupervised Represention Learning for Time Series Based on Signal Mode Decomposition and Ensemble".

**Requirements**

The recommended requirements for SMDE are specified as follows:

- Python==3.8
- torch==1.12.1
- numpy==1.24.4
- pandas==1.4.3
- sktime==0.21.0
- sklearn==1.3.0
- vmdpy==0.2
- matplotlib==3.3.2

**Datasets**

The datasets manipulated in this code can be downloaded on the following locations:

- the UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/;
- the UEA archive: http://www.timeseriesclassification.com/;

**Usage**

To train and evaluate SMDE on the UCR or UEA archives, run the following command:

`python smde_run.py --data_path <data_path> --data_folder <data_folder> --save_path <save_path> --num_imfs <num_imfs> --n_iters <n_iters> --enhance_ways <enhance_ways> --noise_std <noise_std> --use_multi_gpu <use_multi_gpu> --device_ids <device_ids>`

**Hyperparameters**

Hyperparameters are described in smde_run.py.