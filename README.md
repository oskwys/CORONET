![coronet_front_logo.PNG](/coronet_front_logo.PNG)

------

CORONET is an online tool to support decisions regarding hospital admissions or discharge in cancer patients presenting with symptoms of COVID-19 and the likely severity of illness. It is based on real world patient data.

The tool is available at:
https://coronet.manchester.ac.uk/

This repository contains code to run a CORONET model used in the tool to generate recommendation:
- CORONET_functions.py - all functions responsible for data processing, prediction, explanation and finding similar patients in the dataset
- code_testing_CORONET.py - a script to test the main function 'predict_and_explain'
- code_testing_CORONET_functions.py - a script to test individual functions
- most_similar_patients_script.py - a script to test finding similar patients


## Cite
Detailed description of the process of developing CORONET can be found in our publication:

*Establishment of CORONET; COVID-19 Risk in Oncology Evaluation Tool to identify cancer patients at low versus high risk of severe complications of COVID-19 infection upon presentation to hospital*



## Model evolution
- version 1 available at medRxiv. https://www.medrxiv.org/content/10.1101/2020.11.30.20239095v1
- version 2 published in JCO Clinical Cancer Informatics


