# ECG Research

### Description

 Automated detection of various cardiovascular diseases and sleep apnea using a single-lead electrocardiogram

This repository contains the works and research done to develop tools to compute some of the physiological features that could be derived from the electrocardiogram (ECG) signal. 


### Setup

The module uses Python 3 and `virtualenv` to manage the environment. 

Run this from the root of the repo to create the environment:

```
conda env create
source activate ecg
```

### Getting the data

For running the *_dev.nb files for predicting various cardiovascular diseases and sleep apnea, the databases can be found in these links: 

PAC/PVC/BBB Detection [INCARTDB](https://www.physionet.org/pn3/incartdb/)  
VFIB Detection [CUDB](https://physionet.org/physiobank/database/cudb/)  
AFIB Detection [LTAFDB](https://physionet.org/physiobank/database/ltafdb/)  
Sleep Apnea [apnea-ecg](https://www.physionet.org/physiobank/database/apnea-ecg/)  



### How to run the notebooks 

Activate the correct python environment with:

```
source activate ecg
```

Start up the Juptyer (IPython) Notebook:

```
jupyter notebook
```

Running the dev notebooks should build the necessary databases within the local machine in order to run the demo notebooks. 

The database itself can also be downloaded from Xavier Puspus (Cebu Office)


## Authors

* **Maria Eloisa Ventura** - *
* **Xavier Puspus** - *

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.



