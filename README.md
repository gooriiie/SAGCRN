# SAGCRN: Sequence-aware Adaptive Graph Convolutional Networks for Traffic Forecasting  

### Requirements
- python 3.8.16
- pytorch 1.10.1
- pandas 1.5.3
- numpy 1.23.2
- torch-summary 1.4.5

### Datasets
We train and evaluate SAGCRN on three benchmark datasets (METR-LA, PEMS-BAY, and PEMS08).  
The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1Q7Ec6I1i2al_CWt7bPQalYHY4y7e_Tkl?usp=sharing).  
Please put the downloaded dataset into the corresponding directory and then unzip the .zip file.
  
### Run
```shell
cd model
python traintest_SAGCRN.py --dataset METRLA --gpu 0
```
