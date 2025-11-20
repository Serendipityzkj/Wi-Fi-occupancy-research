1. Folder: “Model_predicted_results”
This folder contains the occupancy detection results for eight MAC addresses during the experiment. The data include ground-truth labels, predicted labels from the CNN-GBC model, and predicted labels from the CNN-GBC-RF model.

2. Folder: “Sensor_data_during_experiment”
This folder stores all data recorded by the IoT sensors throughout the experimental period. The dataset includes AC setpoints, cumulative AC energy consumption, workstation illuminance, light status, socket status, wireless-switch feedback, workstation temperature, thermal-comfort questionnaires, and outdoor meteorological data.

3. Folder: “Wi-Fi_signal_data_during_experiment”
This folder contains all Wi-Fi signal data collected by the Wi-Fi probes during the experiment.

4. Files: “ac_energy_consumption_process.py” and “workstation_temp_process.py”
These two scripts demonstrate the processing procedures for cumulative AC energy consumption data and workstation temperature data, respectively, and generate example plots based on the processed results.