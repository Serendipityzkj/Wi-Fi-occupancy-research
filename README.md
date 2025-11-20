# Wi-Fi-occupancy-research
This repository contains the experimental data mentioned in the research paper as well as the relevant code for model development.

1. Dataset during experiment
This folder contains dataset during experiments (10 days for the OCC scenario and 6 days for the Baseline scenario). The experimental data mainly includes:
Air conditioning setpoint parameters; Air conditioning energy consumption data; Indoor temperature data; Desktop illuminance data; Light switch state; Smart socket state; Wireless switch state; Thermal comfort questionnaire data; Outdoor meteorological data; Raw Wi-Fi signal data; Output predicted by models

2. CNN–GBC Model Development
This folder contains the training and testing procedures for both the CNN model and the Gradient Boosting Classifier (GBC). It includes the datasets used for model training as well as the corresponding code.
A detailed description of the CNN–GBC model development workflow is provided in the file readme.txt within this folder.

3. RF Model Development
This folder contains the complete training workflow of the Random Forest (RF) model, including the training dataset and all scripts used for model training and evaluation.

4. Occupancy Model Application
This folder contains the script Prediction.py, which loads the saved CNN, GBC, and RF models and uses the file Testing_samples.xlsx to demonstrate occupancy detection based on Wi-Fi signals.
