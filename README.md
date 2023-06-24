# Clasificación de Triage en mediante Modelos de Aprendizaje Automático
En todas las entidades prestadoras de salud, garantizar la correcta clasificación de los pacientes que acuden a los servicios de urgencias representa un desafío crucial. La adecuada clasificación de triage es fundamental para asegurar que cada paciente reciba la atención adecuada en el momento preciso, sobre todo en situaciones críticas donde cada minuto cuenta.

En este contexto, es una necesidad ayudar a mejorar el proceso de clasificación de triage en salas de urgencias. Esto implica la implementación de herramientas y enfoques que permitan a los profesionales de la salud tomar decisiones más precisas y rápidas al clasificar a los pacientes.

## Objetivo
En ese orden de ideas, el objetivo de este articulo es desarrollar un modelo de clasificación que permita clasificar a los pacientes que llegan a una sala de urgencias, teniendo en cuenta características como su temperatura, oximetría, edad entre otras en conjunto con el texto registrado por el profesional de la salud al momento de atender a los pacientes.

## Métricas y evaluación
Teniendo en cuenta que se busca clasificar a los pacientes en 5 categorías de triage, estamos frente a un problema de clasificación multiclase. Como las clases no están balanceadas, la métrica que se usara para evaluar el desempeño de los modelos será f1 macro score, pues esta pondera la precisión y la exhaustividad.

## Requerimientos:
Este proyecto se realiso en Python con las siguientes librerias: 
* re
* pandas 
* numpy 
* matplotlib.pyplot as plt
* seaborn as sns
* sklearn
* pickle
* nltk

## Descripción de los datos
Para el desarrollo de este proyecto se tiene un conjunto de 20.000 registros de pacientes que han asistido a urgencias entre el año 2021 y 2022. La información esta divida en 2 conjuntos de datos:

base_datos: Base de datos de pacientes con las siguientes variables:
* ID_TM_Afiliado_cod: Id único de cada paciente.
* Edad: Edad del paciente.
* Genero: Genero del paciente.
* SedeAtencion: Luegar de atención.
* TipoAtencion: Tipo de atención recibida.
* TipoAfiliado: Tipo de afiliación del paciente.

2. base_consulta: Base de datos de las consultas con las siguientes variables:

* ID_TM_Afiliado_cod: Id único de cada paciente.
* IdConsulta_cod: ID único de la consulta.
* Fecha: Fecha de la consulta.
* Dia: Día de la consulta.
* Mes: Mes de la consulta.
* AÑO: Año de la consulta.
* TAS: Tensión arterial sistólica.
* TAD: Tensión arterial diastólica.
* TAM: Tensión arterial media.
* FC: Frecuencia cardiaca.
* FR: Frecuencia respiratoria.
* Temp: Temperatura.
* Oximetria: Oximetria.
* Anotacion TRIAGE Consulta: Texto registrado por el especialista.
* ClasificaTriage: Triage asignado al paciente.

## Exploración de datos

Se analizaron los conjuntos de datos que contienen información relevante para la clasificación de triaje en las salas de urgencias. Estos conjuntos de datos incluyen características como temperatura, oximetría, edad y texto registrado por los profesionales de la salud al atender a los pacientes.

Se examinaron las distribuciones de los datos y se calcularon estadísticas descriptivas para identificar cualquier anomalía o característica específica presente en los conjuntos de datos.

## Visualización de datos

Se construyeron visualizaciones de datos basadas en el análisis realizado en el paso anterior. Estas visualizaciones permitieron identificar patrones, tendencias o relaciones entre las características y las clases de triaje. Se utilizaron gráficos, como histogramas y diagramas de caja, para visualizar las distribuciones y relaciones de los datos.

## Tercera fase: Metodología

En esta fase se detallan los pasos llevados a cabo para implementar el modelo de clasificación y refinarlo. A continuación se describe la metodología utilizada:

## Preprocesamiento de datos

Se realizaron etapas de preprocesamiento de datos para asegurar la calidad y consistencia de los mismos. Esto incluyó el manejo de valores atípicos, la normalización de los datos numéricos y la codificación de las variables categóricas. También se realizó un análisis de texto para transformar el texto registrado por los profesionales de la salud en características numéricas que pudieran ser utilizadas por el modelo.

## Implementación

Se seleccionaron tres algoritmos de clasificación multiclase: Regresión Logística, Random Forest y Red Neuronal. Se implementaron pipelines que incluyeron las etapas de preprocesamiento de datos y se aplicó la técnica de grid search para optimizar los parámetros de cada algoritmo. Se ejecutaron tres fases de entrenamiento y evaluación: la primera utilizando solo datos numéricos, la segunda utilizando solo datos de texto y la tercera utilizando datos mixtos.

## Refinamiento

Durante la fase de refinamiento, se realizaron ajustes adicionales en los algoritmos y técnicas utilizados. Esto incluyó la utilización de técnicas de validación cruzada para evaluar el rendimiento de los modelos, así como la modificación de los parámetros de los algoritmos para obtener mejores resultados.

## Resultados 

1. Se llevó a cabo la evaluación y validación de los modelos desarrollados durante la metodología. Se utilizaron diferentes métricas para evaluar el rendimiento de los modelos, como la precisión, la exhaustividad y el valor F1. Además, se compararon los resultados obtenidos por cada modelo en forma de tablas.
2. Se obtiene el modelo pkl adjunto en la carpeta.
3. Articulo en Medium: https://medium.com/@diego.maca/clasificaci%C3%B3n-de-triage-en-mediante-modelos-de-aprendizaje-autom%C3%A1tico-5491976aa091

## Conclusiones
En resumen, este proyecto abordó el desafío de la clasificación de triage en los servicios de salud. Mediante el uso de diversos modelos y técnicas, desarrollamos una solución integral que brinda apoyo a los profesionales de la salud para clasificar con precisión a los pacientes.

En cuanto a mejoras: Se sugiere continuar la investigación para mejorar el experimento. Es importante destacar que el modelo desarrollado está listo para ser complementado con una aplicación que facilite su despliegue y logre una solución integral de extremo a extremo. Esta herramienta complementaria tiene como objetivo apoyar y mejorar el trabajo de los médicos en situaciones de urgencia, permitiéndoles tomar decisiones más precisas y rápidas en la clasificación de triage.

## Reconocimiento:
1. Udacity Data scientist course.

## Bibliografia:
1. Choi, S. W., Ko, T., Hong, K. J., & Kim, K. H. (2019). Machine learning-based prediction of Korean triaje and acuity scale level in emergency department patients. Healthcare informatics research, 25(4), 305–312. 10.4258/hir.2019.25.4.305.

## Autor 
Diego Maca
