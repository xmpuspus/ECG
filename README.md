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

Signal Quality (good or bad ECG beat) Detection [signal-quality](https://physionet.org/physiobank/database/challenge/2011/)  
PAC/PVC/BBB Detection [INCARTDB](https://www.physionet.org/pn3/incartdb/)  
VFIB Detection [CUDB](https://physionet.org/physiobank/database/cudb/)  
AFIB Detection [LTAFDB](https://physionet.org/physiobank/database/ltafdb/)  
Sleep Apnea [apnea-ecg](https://www.physionet.org/physiobank/database/apnea-ecg/)  

To download all the data from each link, use this command:  

```
wget -r -np http://www.physionet.org/insert_specific_URL
```


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

The database itself which includes the raw ECG signal and the structured data is around 6GB can also be sourced directly from *Xavier Puspus* (Cebu Office).



### Authors

* **Maria Eloisa Ventura** - ECG Signal Quality and Pre-Processing
* **Xavier Puspus** - Detection of CVDs and Sleep Apnea from ECG


### Acknowledgements

Special thanks to:

* Greg Romrell for the guidance all throughout the development of the modules,
* Joseph Roxas for the tips along the way.




