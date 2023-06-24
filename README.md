# Triage Classification through Machine Learning Models
In all healthcare institutions, ensuring the accurate classification of patients who come to emergency services represents a crucial challenge. Proper triage classification is essential to ensure that each patient receives the appropriate care at the right time, especially in critical situations where every minute counts.

In this context, there is a need to improve the triage classification process in emergency rooms. This involves implementing tools and approaches that allow healthcare professionals to make more precise and faster decisions when classifying patients.

## Objective
In light of this, the objective of this article is to develop a classification model that allows for the classification of patients arriving at an emergency room, taking into account characteristics such as temperature, oxygen saturation, age, among others, along with the text recorded by the healthcare professional when attending to the patients.

## Metrics and Evaluation
Considering that the goal is to classify patients into 5 triage categories, we are facing a multiclass classification problem. Since the classes are not balanced, the metric used to evaluate the performance of the models will be the macro F1 score, as it weighs precision and recall.

## Requirements:
This project was carried out in Python with the following libraries:

* re
* pandas
* numpy
* matplotlib.pyplot as plt
* seaborn as sns
* sklearn
* pickle
* nltk

## Contents
In this post, you can find:

* base_consulta.csv
* base_datos.csv
* Notebook Clasificacion_triage.py
* Notebook Clasificacion_triaje.ipnynb
* base_consulta.zip

 You can unzip the base query folder and there you will find the notebook and the model in pkl format.
 
## Description of the Data
For the development of this project, there is a dataset of 20,000 patient records who have attended emergency rooms between 2021 and 2022. The information is divided into 2 datasets:

1.  base_datos: Patient database with the following variables:

* ID_TM_Afiliado_cod: Unique ID for each patient.
* Edad: Patient's age.
* Genero: Patient's gender.
* SedeAtencion: Place of care.
* TipoAtencion: Type of care received.
* TipoAfiliado: Patient's affiliation type.

2. base_consulta: Consultation database with the following variables:

* ID_TM_Afiliado_cod: Unique ID for each patient.
* IdConsulta_cod: Unique ID for the consultation.
* Fecha: Consultation date.
* Dia: Consultation day.
* Mes: Consultation month.
* AÑO: Consultation year.
* TAS: Systolic blood pressure.
* TAD: Diastolic blood pressure.
* TAM: Mean arterial pressure.
* FC: Heart rate.
* FR: Respiratory rate.
* Temp: Temperature.
* Oximetria: Oxygen saturation.
* Anotacion TRIAGE Consulta: Text recorded by the specialist.
* ClasificaTriage: Triage assigned to the patient.

## Data Exploration
The datasets containing relevant information for triage classification in emergency rooms were analyzed. These datasets include characteristics such as temperature, oxygen saturation, age, and text recorded by healthcare professionals when attending to patients.

The data distributions were examined, and descriptive statistics were calculated to identify any anomalies or specific features present in the datasets.

## Data Visualization
Data visualizations were constructed based on the analysis conducted in the previous step. These visualizations allowed for the identification of patterns, trends, or relationships between the characteristics and triage classes. Graphs such as histograms and box plots were used to visualize the distributions and relationships of the data.


## Data Preprocessing
Data preprocessing stages were performed to ensure the quality and consistency of the data. This included handling outliers, normalizing numerical data, and encoding categorical variables. Text analysis was also conducted to transform the text recorded by healthcare professionals into numerical features that could be used by the model.

## Implementation
Three multiclass classification algorithms were selected: Logistic Regression, Random Forest, and Neural Network. Pipelines were implemented, including data preprocessing stages, and the grid search technique was applied to optimize the parameters of each algorithm. Three training and evaluation phases were executed: the first using only numerical data, the second using only text data, and the third using mixed data.

## Refinement
During the refinement phase, additional adjustments were made to the algorithms and techniques used. This included using cross-validation techniques to evaluate the performance of the models, as well as modifying the parameters of the algorithms to obtain better results.

## Results
1. The evaluation and validation of the models developed during the methodology were conducted. Different metrics, such as accuracy, recall, and F1 score, were used to evaluate the performance of the models. The obtained results for each model were compared in tabular form.
2. The attached model pkl file is included in the folder.
3. Article on Medium: https://medium.com/@diego.maca/clasificaci%C3%B3n-de-triage-en-mediante-modelos-de-aprendizaje-autom%C3%A1tico-5491976aa091

## Conclusions
In summary, this project addressed the challenge of triage classification in healthcare services. Through the use of various models and techniques, we developed a comprehensive solution that supports healthcare professionals in accurately classifying patients.

Regarding improvements, it is suggested to continue research to enhance the experiment. It is important to note that the developed model is ready to be complemented with an application that facilitates its deployment, achieving an end-to-end comprehensive solution. This complementary tool aims to support and improve the work of doctors in emergency situations, enabling them to make more precise and faster triage classification decisions.

## Acknowledgments:
1. Udacity Data Scientist Course.
2. Virrey Solis for providing the data for academic purposes.

## Usage
The data provided for this project was solely for academic and non-commercial purposes. Furthermore, it was anonymized and treated in such a way that sensitive data is not part of the project and not available to the public.

## Bibliography:
1. Choi, S. W., Ko, T., Hong, K. J., & Kim, K. H. (2019). Machine learning-based prediction of Korean triage and acuity scale level in emergency department patients. Healthcare informatics research, 25(4), 305–312. 10.4258/hir.2019.25.4.305.

## Author
Diego Maca






