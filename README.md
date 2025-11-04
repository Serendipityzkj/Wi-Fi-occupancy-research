# Wi-Fi-occupancy-research
This repository contains the experimental data mentioned in the research paper as well as the relevant code for model development.
1. Experimental Data
This folder contains merged experimental data (10 days for the OCC scenario and 6 days for the Baseline scenario). The experimental data mainly includes:
	Air conditioning setpoint parameters
	Air conditioning energy consumption data
	Indoor temperature data
	Desktop illuminance data
	Light switch state
	Smart socket state
	Wireless switch state
	Thermal comfort questionnaire data
	Outdoor meteorological data 
	Raw Wi-Fi signal data
	Output predicted by models
2. Model Training
Model training mainly involves three models: CNN, GBC, and RF. This folder includes the data used for model training as well as the code for training and testing the models.
1) Model Training Data
	Input_data_CNN.csv: This file is used for training and testing CNN models. It contains the collected original Wi-Fi signal data and the real occupancy labels.
	GBC data: The data inside this folder is used to train the GBC model. This data is the output from the trained CNN model, and the training set and test set are split into an 8:2 ratio manually.
	Input_data_RF.csv: This file is used to train the RF model. The data is real operation data and is a sample set selected from a single individual.
2) Model Training Code 
The files CNN_model.ipynb, GBC_model.ipynb, and RF_model.ipynb are used for the model training that can be run on the Jupyter Notebook application, while the files CNN_model.html, GBC_model.html, RF_model.html are the display of the output results of the code running.
