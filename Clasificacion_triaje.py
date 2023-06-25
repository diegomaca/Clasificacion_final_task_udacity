#!/usr/bin/env python
# coding: utf-8

# # Triage Classification through Machine Learning Models
# In all healthcare institutions, ensuring the accurate classification of patients who come to emergency services represents a crucial challenge. Proper triage classification is essential to ensure that each patient receives the appropriate care at the right time, especially in critical situations where every minute counts.
# 
# In this context, there is a need to improve the triage classification process in emergency rooms. This involves implementing tools and approaches that allow healthcare professionals to make more precise and faster decisions when classifying patients.
# 
# ## Objective
# In light of this, the objective of this article is to develop a classification model that allows for the classification of patients arriving at an emergency room, taking into account characteristics such as temperature, oxygen saturation, age, among others, along with the text recorded by the healthcare professional when attending to the patients.
# 
# ## Metrics and Evaluation
# Considering that the goal is to classify patients into 5 triage categories, we are facing a multiclass classification problem. Since the classes are not balanced, the metric used to evaluate the performance of the models will be the macro F1 score, as it weighs precision and recall.
# 

# ## Requirements:
# This project was carried out in Python with the following libraries:

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[3]:


from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))


# In[4]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import pickle


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# ## Description of the Data
# For the development of this project, there is a dataset of 20,000 patient records who have attended emergency rooms between 2021 and 2022. The information is divided into 2 datasets:
# 
# 1.  base_datos: Patient database with the following variables:
# 
# * ID_TM_Afiliado_cod: Unique ID for each patient.
# * Edad: Patient's age.
# * Genero: Patient's gender.
# * SedeAtencion: Place of care.
# * TipoAtencion: Type of care received.
# * TipoAfiliado: Patient's affiliation type.
# 
# 2. base_consulta: Consultation database with the following variables:
# 
# * ID_TM_Afiliado_cod: Unique ID for each patient.
# * IdConsulta_cod: Unique ID for the consultation.
# * Fecha: Consultation date.
# * Dia: Consultation day.
# * Mes: Consultation month.
# * AÑO: Consultation year.
# * TAS: Systolic blood pressure.
# * TAD: Diastolic blood pressure.
# * TAM: Mean arterial pressure.
# * FC: Heart rate.
# * FR: Respiratory rate.
# * Temp: Temperature.
# * Oximetria: Oxygen saturation.
# * Anotacion TRIAGE Consulta: Text recorded by the specialist.
# * ClasificaTriage: Triage assigned to the patient.
# 

# In[6]:


url_base_datos = 'base_datos.csv'
url_base_consulta = 'base_consulta.csv'


# In[7]:


base_datos = pd.read_csv(url_base_datos).drop('Unnamed: 0', axis = 1)
base_consulta = pd.read_csv(url_base_consulta).drop('Unnamed: 0', axis = 1)


# ## Data Exploration
# The datasets containing relevant information for triage classification in emergency rooms were analyzed. These datasets include characteristics such as temperature, oxygen saturation, age, and text recorded by healthcare professionals when attending to patients.
# 
# The data distributions were examined, and descriptive statistics were calculated to identify any anomalies or specific features present in the datasets.
# 
# ### Base_datos

# In[8]:


base_datos.head(2)


# In[9]:


# Nro de registros y columnas
base_datos.shape


# In[10]:


# Nro duplicados
base_datos.duplicated().sum()


# In[11]:


base_datos.drop_duplicates(inplace = True)


# In[12]:


base_datos.nunique()


# In[13]:


# Valores faltantes
base_datos.isna().sum()


# In[14]:


def comp_grap(var1, x_label1, y_label1, title1, title2 ):
    """
    Creates a comparative graph of a numeric variable using a boxplot and a histogram.

    Arguments:
    - var1: numeric variable to create the comparative graph.
    - x_label1: x-axis label of the boxplot and histogram.
    - y_label1: label of the y axis of the boxplot.
    - title1: title of the boxplot.
    - title2: histogram title.

    Returns:
    - None: shows the comparative graph.
    """

    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5))
    #plt.figure(figsize=(8,12))
    ax1.boxplot(var1)
    ax1.set_xlabel(x_label1)
    ax1.set_ylabel(y_label1)
    ax1.set_title(title1)

    ax2.hist(var1, bins = 30)
    ax2.set_xlabel(x_label1)
    ax2.set_ylabel('Conteo')
    ax2.set_title(title2)

    plt.tight_layout()

    plt.show()


# In[15]:


base_datos.Edad.describe()


# In[16]:


comp_grap(base_datos.Edad, 'Edad', 'Edad en años', 'Box plot Edad', 'Distribución de la edad')


# The description of age reveals that the count of data is 19,407 observations. The mean age is approximately 31.2 years, with a standard deviation of around 15.8 years. The minimum recorded value is 0 years, indicating the presence of outliers or data collection errors. 25% of the recorded ages are below 22 years, while 50% are below 30 years. 75% of the ages fall below 40 years, indicating a distribution skewed towards younger ages. The maximum recorded age is 97 years, suggesting the presence of a smaller group of older individuals in the dataset.

# In[17]:


def bar_graps(var, x_label, y_label, title, rot):
    """
    Arguments:

    -var: categorical variable to create the bar chart.
    -x_label: label for the x-axis.
    -y_label: label for the y-axis.
    -title: title of the chart.
    -rot: rotation angle of the x-axis labels (in degrees).
    Returns:

    -None: displays the bar chart.
    """
    plt.bar(var.value_counts().index.to_list(), var.value_counts().to_list() )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=rot)
    plt.show()


# In[18]:


# Distribución de lugar de atención
bar_graps(base_datos.SedeAtencion, 'Sede de atención', 'Valores', 'Gráfico de barras de sede de atención', 45)


# In[19]:


base_datos.SedeAtencion.value_counts(normalize = True)


# These values represent percentages of attention distribution across different locations. UUBC Las Américas accounts for 47.05% of the total attention, making it the location with the highest proportion. UUBC Calle 98 follows with 30.80% of the attention, indicating a slightly lower percentage. UUBC Soacha represents 20.27% of the attention, while UUBC Americas 2 has the lowest percentage at 1.88%. These figures highlight the varying proportions of attention allocated to each respective location.

# In[20]:


# Distribución de lugar de atención
bar_graps(base_datos.Genero, 'Genero', 'Valores', 'Gráfico de barras de Genero', 0)


# In[21]:


base_datos.Genero.value_counts(normalize = True)


# The percentage breakdown of the patient population by gender reveals that 56.07% are female, while 43.93% are male. This indicates that a larger proportion of the patients are female, accounting for over half of the total population, while males make up the remaining percentage.

# In[22]:


# Distribución de lugar de atención
bar_graps(base_datos.TipoAfiliado, 'Tipo afiliación', 'Valores', 'Gráfico de barras de Tipo afiliación', 0)


# In[23]:


base_datos.TipoAfiliado.value_counts(normalize = True)


# These values pertain to the type of affiliation among individuals. The breakdown is as follows: 69.26% are classified as "Cotizante" (contributor), 30.72% are "Beneficiario" (beneficiary), and a very small proportion of 0.03% are labeled as "Inexistente" (non-existent). These percentages provide insights into the distribution of individuals based on their affiliation types within the dataset.

# ### Base de consultas

# In[24]:


base_consulta.head(2)


# In[25]:


# Nro de registros y columnas
base_consulta.shape


# In[26]:


# Nro duplicados
base_consulta.duplicated().sum()


# In[27]:


# Nro valores unicos por columna
base_consulta.nunique()


# In[28]:


# Valores faltantes
base_consulta.isna().sum()


# In[29]:


# Distribución de lugar de atención
bar_graps(base_consulta.ClasificaTriage , 'Etiqueta Triage', 'Valores', 'Gráfico de barras de Clasificación Triage', 0)


# In[30]:


base_consulta.ClasificaTriage.value_counts(normalize = True)


# These values represent the distribution of consultations across different Triage categories. It's important to note that there are 5 categories, with 1 being the most critical and 5 being the least critical. Here is the breakdown along with the corresponding percentages:
# 
# * Category 3: 31.00%
# * Category 2: 30.10%
# * Category 4: 27.14%
# * Category 1: 11.58%
# * Category 0: 0.18%
# 
# It is worth mentioning that **Category 1 will not be used** in the analysis as it represents less than 1% of the data. These percentages provide insights into the relative distribution of consultations across the Triage categories, with Categories 3 and 2 being the most commonly observed.

# In[31]:


base_consulta['TAM'] = base_consulta['TAM'].str.replace(',', '.').astype(float)


# In[32]:


# Obtener la matriz de correlación
corr_matrix = base_consulta [['TAS', 'TAD', 'TAM', 'FC', 'FR', 'Temp', 'Oximetria']].corr()

# Crear el mapa de calor
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Personalizar el mapa de calor
plt.title('Matriz de correlación')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Mostrar el mapa de calor
plt.show()


# ### Reamrk
# 
# The correlation matrix reveals the correlations among different variables. It's important to note that the variables TAS (Systolic Blood Pressure), TAD (Diastolic Blood Pressure), and TAM (Mean Arterial Pressure) exhibit high correlation coefficients. Therefore, we will only use TAM, as it is derived from both TAD and TAS.
# 
# Based on the high correlation between TAS, TAD, and TAM, we will use TAM as the representative variable.

# In[33]:


comp_grap(base_consulta.TAM, 'TAM', 'Escala TAM', 'Box plot TAM', 'Distribución de TAM')


# In[34]:


base_consulta.TAM.describe()


# The descriptive statistics for TAM (Mean Arterial Pressure) are as follows: The mean value is 82.487433, with a standard deviation of 29.482035. The minimum recorded value is 0.000000, indicating the presence of potential outliers or measurement errors. The 25th percentile is 82.000000, meaning that 25% of the data falls below this value. The median (50th percentile) is 89.330000, representing the midpoint of the data distribution. The 75th percentile is 96.000000, indicating that 75% of the data falls below this value. The maximum recorded value is 202.000000, highlighting the presence of potentially extreme measurements. These statistics provide a summary of the distribution and variability of the Mean Arterial Pressure values.

# In[35]:


comp_grap(base_consulta.FC, 'FC', 'Escala FC', 'Box plot FC', 'Distribución de FC')


# In[36]:


base_consulta.FC.describe()


# The descriptive statistics for Heart Rate (FC) are as follows: The mean value is 87.850500, with a standard deviation of 19.922269. The minimum recorded value is 0.000000, indicating potential outliers or measurement errors. The 25th percentile is 75.000000, meaning that 25% of the data falls below this value. The median (50th percentile) is 84.000000, representing the midpoint of the data distribution. The 75th percentile is 97.000000, indicating that 75% of the data falls below this value. The maximum recorded value is 210.000000, suggesting the presence of potentially extreme measurements. These statistics provide a summary of the distribution and variability of Heart Rate values.

# In[37]:


comp_grap(base_consulta.FR, 'FR', 'Escala FR', 'Box plot FR', 'Distribución de FR')


# In[38]:


base_consulta.FR.describe()


# The descriptive statistics for Respiratory Rate (FR) are as follows: The mean value is 19.168950, with a standard deviation of 5.566584. The minimum recorded value is 0.000000, suggesting potential outliers or measurement errors. The 25th percentile is 18.000000, indicating that 25% of the data falls below this value. Both the median (50th percentile) and the 75th percentile are 18.000000, suggesting that the data is concentrated around this value. The maximum recorded value is 100.000000, indicating the presence of potentially extreme measurements. These statistics provide a summary of the distribution and variability of Respiratory Rate values.

# In[39]:


comp_grap(base_consulta.Oximetria, 'Oximetria', 'Escala Oximetria', 'Box plot Oximetria', 'Distribución de Oximetria')


# In[40]:


base_consulta.Oximetria.describe()


# ### Remark 
# 
# Finally, it is worth mentioning that there are many outliers observed in the vital signs variables. However, it is important to note that these data points are from an emergency room setting, where the presence of outliers is expected. Therefore, **no outlier data will be removed from the dataset**, as they are considered valuable and representative of the real-world scenarios encountered in an emergency healthcare environment.

# In[41]:


# Box plot de la variable edad
fig, (ax1,ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))
#plt.figure(figsize=(8,12))
ax1.bar(base_consulta.groupby('DIA')['DIA'].count().sort_values(ascending = False).index.to_list(), base_consulta.groupby('DIA')['DIA'].count().sort_values(ascending = False).values )
ax1.set_xlabel('Dia mes')
ax1.set_ylabel('conteo')
ax1.set_title('Consultas por dia')

ax2.bar(base_consulta.groupby('MES')['MES'].count().sort_values(ascending = False).index.to_list(), base_consulta.groupby('MES')['MES'].count().sort_values(ascending = False).values )
ax2.set_xlabel('Mes')
ax2.set_ylabel('Conteo')
ax2.set_title('Consultas por mes')

ax3.bar(['2021','2022'], base_consulta.groupby('AÑO')['AÑO'].count().sort_values(ascending = False).values )
ax3.set_xlabel('Año')
ax3.set_ylabel('Conteo')
ax3.set_title('Consultas por año')

plt.tight_layout()

plt.show()


# In[42]:


base_consulta.groupby('AÑO')['AÑO'].count()/20000


# In[43]:


base_consulta.groupby('DIA')['DIA'].count().sort_values(ascending = False)/20000


# In[44]:


base_consulta.groupby('MES')['MES'].count().sort_values(ascending = False)/20000


# On the other hand, more than 51% of the attentions were given in the first half of the year. In addition, the days with the least number of queries are generally in the last days of the month, as shown below.

# ## Data Preprocessing
# Data preprocessing stages were performed to ensure the quality and consistency of the data. This included handling outliers, normalizing numerical data, and encoding categorical variables. Text analysis was also conducted to transform the text recorded by healthcare professionals into numerical features that could be used by the model.
# 

# In[45]:


base = base_datos.merge(base_consulta, on = ['ID_TM_Afiliado_cod'], how = 'left').drop_duplicates(['ID_TM_Afiliado_cod', 'IdConsulta_cod'])


# In[46]:


base.nunique()


# In[47]:


base.head(2)


# In[48]:


base['ClasificaTriage'].value_counts(normalize = True )


# In[49]:


# Filtro clase diferente a 0
base = base.query('ClasificaTriage > 0')


# In[50]:


base.isna().sum()


# In[51]:


base['Temp']= base['Temp'].fillna(base['Temp'].mean())
#base = base.dropna()


# In[52]:


base['ClasificaTriage'].value_counts( )


# ## Cleaning

# In[53]:


def limpieza(text):
  """
    Performs text cleanup by removing non-alphanumeric characters and converting them to lowercase.

    Arguments:
    - text: text to clean.

    Returns:
    - clean_text: clean text without non-alphanumeric characters and in lower case.
    """
  text_limpio = re.sub('[\W]+', ' ', text.lower())
  return text_limpio


# In[54]:


base['final'] = base['Anotacion TRIAGE Consulta'].apply(limpieza).apply(str)


# # Feature Selection
# ## Number of words
# 
# Keywords were determined using the tf-idf model, comparing the variance explained by different subsets of words.

# In[55]:


my_stopwords = stopwords.words('spanish')


# In[56]:


tfidf = TfidfVectorizer(stop_words= my_stopwords)


# In[57]:


docs = base['final'].values


# In[58]:


X = tfidf.fit_transform(docs)


# In[59]:


X.shape


# In[60]:


variance = []
for i in [100, 200, 300, 400,500, 600, 700, 800, 900,1000]:
    tfidf_temp = TfidfVectorizer(stop_words= my_stopwords, max_features=i)
    X_temp = tfidf_temp.fit_transform(docs)
    variance.append(X_temp.sum()/X.sum())
    print(i, X_temp.sum()/X.sum())


# In[61]:


plt.xlabel('Número de palabras')
plt.ylabel('Varianza explicada')
plt.plot([100, 200, 300, 400,500, 600, 700, 800, 900,1000], variance)
plt.axvline(x=800, linestyle='--', color='r') # Línea vertical en 500
plt.hlines(y=0.90, xmin=0, xmax=1000, linestyle='--', color='r') # Línea horizontal en 0.95
plt.show()


# Note that, from a set of 22,635 different words in the text corpus, the 800 most significant words are selected. As can be seen in the graph, with this amount of words, approximately 90% of the information is captured in the text present in the database.

# ## Preparation of text data

# In[62]:


# Base solo texto
X_texto = base['final'].values
y_texto= base['ClasificaTriage'].values


# In[63]:


my_stopwords = stopwords.words('spanish')


# In[64]:


vectorizer_texto = TfidfVectorizer(stop_words= my_stopwords)
tfidf_matrix = vectorizer_texto.fit_transform(X_texto)


# In[65]:


def get_most_important_words_index(tfidf_matrix, vectorizer, n=10):
    """
    Devuelve los índices de las n palabras más importantes en la matriz TF-IDF
    """
    # Obtiene el vector de características que representa el vocabulario
    feature_names = vectorizer.vocabulary_.keys()
    feature_names = list(feature_names)

    # Obtiene la puntuación media de TF-IDF para cada palabra en la matriz TF-IDF
    mean_tfidf_scores = tfidf_matrix.mean(axis=0)

    # Crea una lista de tuplas (índice de la palabra, puntuación TF-IDF media)
    word_scores = [(col, mean_tfidf_scores[0, col]) for col in range(len(list(set(tfidf_matrix.nonzero()[1]))))]

    # Ordena la lista de palabras por la puntuación TF-IDF media en orden descendente
    sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

    # Obtiene los índices de las n palabras más importantes
    top_word_indices = [x[0] for x in sorted_word_scores[:n]]

    return top_word_indices


# In[66]:


important_words_index = get_most_important_words_index(tfidf_matrix, vectorizer_texto, n=800)


# In[67]:


# Selecciona las columnas correspondientes a las palabras más importantes
X_texto_reduced = tfidf_matrix[:, important_words_index]
X_texto_reduced.shape


# In[68]:


# Obtenemos los nombres de las palabras
palabras = vectorizer_texto.vocabulary_.keys()

# Calculamos el puntaje TF-IDF promedio para cada palabra
puntajes = tfidf_matrix.mean(axis=0).tolist()[0]

# Creamos una lista de tuplas (palabra, puntaje) y ordenamos por puntaje
palabras_puntajes = list(zip(palabras, puntajes))
palabras_puntajes.sort(key=lambda x: x[1], reverse=True)

print("Top 10 most important words:")
for palabra, puntaje in palabras_puntajes[:10]:
    print("- {} ({:.4f})".format(palabra, puntaje))


# ## Important numerical and categorical variables
# 
# This selection is done by first identifying which variables have the strongest relationship with the target variable, using the python selectKbest module.

# In[69]:


base_numerica = base[['Edad', 'Genero',  'ClasificaTriage', 'SedeAtencion' , 'TipoAfiliado',  'TAM','FC', 'FR', 'Temp', 'Oximetria']]


# In[70]:


X_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)
y_num= base_numerica['ClasificaTriage'].values


# In[71]:


#k_best = SelectKBest(f_classif, k=6)
#X_new = k_best.fit_transform(X_num, y_num)
#X_num = X_num.iloc[:,list(k_best.get_support())]


# In[72]:


scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)


# In[73]:


X_num


# In[74]:


def importancia_variables_modelo(df, y, modelo):

    scaler = StandardScaler()
    scaler.fit(df)
    X_scaled = scaler.transform(df)

    X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_scaled, y_num, test_size=0.2, random_state=42)

    var = []

    for i in range(1,df.shape[1]):
      selector = SelectKBest(f_classif, k=i)
      X_train_selected = selector.fit_transform(X_num_train, y_num_train)

      model = modelo
      modelo.fit(X_train_selected, y_num_train)

      selected_features_score = modelo.score(X_train_selected, y_num_train)

      model_all_features = modelo
      model_all_features.fit(X_num_train, y_num_train)

      all_features_score = model_all_features.score(X_num_train, y_num_train)

      var.append(selected_features_score/all_features_score)

      #print(f"La variabilidad explicada por las {i:.2f} características seleccionadas es: {(selected_features_score/all_features_score):.2f} %")

    return var


# In[75]:


var_random_forest = importancia_variables_modelo(X_num, y_num, RandomForestClassifier())


# In[76]:


var_logistic_reg = importancia_variables_modelo(X_num, y_num, LogisticRegression())


# In[77]:


var_red = importancia_variables_modelo(X_num, y_num, MLPClassifier())


# ### Remark 
# 
# On the other hand, in order to select the number of variables to select, the analysis of the performance of base models such as logistic regression and random forest trained with all the variables is performed compared to performance if we train it only with a subset of the variables as observed in the graph. In conclusion, we see that with 6 variables we achieve 90% performance in the models compared to training with all the variables.

# In[78]:


x = range(1, X_num.shape[1])
plt.plot(x, var_red, 'bo-', label='VAR Red')
plt.plot(x, var_logistic_reg, 'go-', label='VAR Logistic Regression')
plt.plot(x, var_random_forest, 'ro-', label='VAR Random Forest')



# Agregar leyenda y títulos
plt.legend()

# Línea punteada en x = 6
plt.axvline(x=6, color='k', linestyle='--', label='x = 6')

# Línea punteada en y = 0.9
plt.axhline(y=0.9, color='k', linestyle='--', label='y = 0.9')

plt.title('Comparación de Variables')
plt.xlabel('Número de Variables')
plt.ylabel('Valor')

# Mostrar la gráfica
plt.show()


# ## Preparation of numerical data

# In[79]:


X_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)

k_best = SelectKBest(f_classif, k=6)
X_new = k_best.fit_transform(X_num, y_num)
X_num = X_num.iloc[:,list(k_best.get_support())]


# In[80]:


X_num.shape


# In[81]:


#X_num = X_num[:,k_best.get_support()]


# In[82]:


mask = k_best.get_support()


# In[83]:


base_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)


# In[84]:


fvalues, pvalues = f_classif(base_num.iloc[:, mask], y_num)


# ### Remark 
# 
# For the selection of the most important numerical variables, the relationship of each variable with the objective variable is measured through correlation, resulting in the 6 most important variables in descending order:

# In[85]:


df = pd.DataFrame({'Característica': base_num.columns[mask], 'F': fvalues, 'p-valor': pvalues})
df.sort_values('F')


# ## Development modeling phase
# 
# In this notebook a complete analysis of the training of three algorithms to solve a multiclass problem is presented.
# 
# The first phase is the training of the algorithms using only the numerical data, using the grid search methodology to search for the best hyperparameters and their corresponding evaluation. The metrics obtained for each model are presented.
# 
# The second phase repeats the previous process, but including the text data. It shows how the models improve by incorporating this additional information and the results are compared with the previous phase.
# 
# In the third phase, a joint training is carried out with the numerical and text data. The grid search methodology is used again and the resulting models are evaluated. In addition, the most important features are extracted and the relevant results of the selected model are presented.
# 
# In summary, this part of the project provides a complete vision of the process of training several algorithms to solve a multiclass problem, using different methodologies and data. The metrics and results obtained in each phase are presented, which allows comparing the different models and determining which is the most appropriate for the problem.
# 
# 
# 
# 

# # Numerical modeling
# ## Definition of the metric to optimize

# In[86]:


def evaluate_model(model,x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)

    clsReport = classification_report( y_test,y_pred )
    return clsReport


# In[87]:


X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_scaled, y_num, test_size=0.2, random_state=42)


# In[88]:


scorer = make_scorer(f1_score, average='macro')


# ## Logistic Regression

# In[89]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Transformador: StandardScaler
    ('classifier_lr', LogisticRegression())
])

parameters = {
    'scaler__with_mean': [True, False],
    'classifier_lr__C': [0.1, 1.0]
}

cv_reg = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_reg.fit(X_num_train, y_num_train)


# In[90]:


print(evaluate_model(cv_reg,X_num_test, y_num_test))


# ## Random Forest

# In[91]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Transformador: StandardScaler
    ('classifier_rf', RandomForestClassifier())
])

parameters = {
    'scaler__with_mean': [True, False],
    'classifier_rf__n_estimators': [50,100]
}

cv_rf = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_rf.fit(X_num_train, y_num_train)


# In[92]:


print(evaluate_model(cv_rf,X_num_test, y_num_test))


# ## Neuronal Network

# In[93]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Transformador: StandardScaler
    ('classifier_rn', MLPClassifier())
])

parameters = {
    'scaler__with_mean': [True, False],
    'classifier_rn__hidden_layer_sizes': [(50,), (100,)]
}

cv_rn = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_rn.fit(X_num_train, y_num_train)


# In[94]:


print(evaluate_model(cv_rn,X_num_test, y_num_test))


# ### Remark
# In the numerical phase, the best model was a neural network. Although this model was able to adequately classify some categories, it is observed that the general metrics are moderate, indicating the need for improvements.
# 
# 

# # Text shaping
# ## Logistic regression

# In[95]:


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


# In[96]:


X_texto = base['Anotacion TRIAGE Consulta'].values
y_texto = base['ClasificaTriage'].values


# In[97]:


X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_texto, y_texto, test_size=0.2, random_state=42)


# In[98]:


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words = my_stopwords, max_features= 800)),
    ('classifier', LogisticRegression())
])

parameters = {
    'tfidf__use_idf': (True, False),
    'classifier__C': [0.1, 1.0]
}

cv_reg_text = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_reg_text.fit(X_text_train, y_text_train)


# In[99]:


print(evaluate_model(cv_reg_text,X_text_test, y_text_test))


# ## Random Forest

# In[100]:


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words = my_stopwords, max_features= 800)),
    ('classifier', RandomForestClassifier())
])

parameters = {
    'tfidf__use_idf': (True, False),
    'classifier__n_estimators': [50, 100]
}

cv_rf_text = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_rf_text.fit(X_text_train, y_text_train)


# In[101]:


print(evaluate_model(cv_rf_text,X_text_test, y_text_test))


# ## Neuronal network

# In[102]:


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words = my_stopwords, max_features= 800)),
    ('classifier', MLPClassifier())
])

parameters = {
    'tfidf__use_idf': (True, False),
    'classifier__hidden_layer_sizes': [(50,), (100,)]
}

cv_rn_text = GridSearchCV(pipeline, param_grid=parameters, cv= 3, n_jobs = -1, scoring = scorer )

cv_rn_text.fit(X_text_train, y_text_train)


# In[103]:


print(evaluate_model(cv_rn_text,X_text_test, y_text_test))


# ### Remark 
# 
# In the text phase, logistic regression stood out as the best model, showing better performance compared to the numerical phase.
# These metrics show a significant improvement in classification ability, especially in terms of accuracy and f1-score.

# # Mixto

# ## Data preparation
# 
# An additional step before the training phase with mixed data is to prepare the data, since since structured and text data are mixed, we must correctly define the pipelines, it is important to take into account the type of structure that the data has for which it is used. a transformer that distinguishes the columns.

# In[104]:


def split_dataframe(df1):
    df1 = pd.get_dummies(df1, columns = ['Genero', 'SedeAtencion', 'TipoAfiliado'],  drop_first = True)
    df1 = df1[['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante', 'Anotacion TRIAGE Consulta', 'ClasificaTriage']]
    #X_texto = df1['Anotacion TRIAGE Consulta'].values
    return df1


# In[105]:


split_dataframe(base)


# In[106]:


# Desde aqui
X = split_dataframe(base)[['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante', 'Anotacion TRIAGE Consulta']]
y = split_dataframe(base)['ClasificaTriage']
X_mixto_train, X_mixto_test, y_mixto_train, y_mixto_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[107]:


# Crear el ColumnTransformer para encadenar los transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), ['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante']),
        ('text', TfidfVectorizer(tokenizer=tokenize, stop_words = my_stopwords, max_features= 800), 'Anotacion TRIAGE Consulta')
    ])


# ### Lienar regression

# In[108]:


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression())
])

param_grid = {
    'preprocess__text__ngram_range': [(1, 1), (1, 2)],  # Ejemplo de parámetro para TfidfVectorizer
    'model__C': [0.1, 1, 10]  # Ejemplo de parámetro para LogisticRegression
}

# Crear el objeto GridSearchCV
grid_search_lr = GridSearchCV(pipeline, param_grid, cv=3, scoring=scorer)

# Ajustar el GridSearchCV utilizando los datos de entrenamiento y las etiquetas
grid_search_lr.fit(X_mixto_train, y_mixto_train)


# In[109]:


print(evaluate_model(grid_search_lr,X_mixto_test, y_mixto_test))


# ### Random Forest

# In[110]:


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier())
])

param_grid = {
    'preprocess__text__ngram_range': [(1, 1), (1, 2)],  # Ejemplo de parámetro para TfidfVectorizer
    'model__n_estimators': [50, 100]  # Ejemplo de parámetro para LogisticRegression
}

# Crear el objeto GridSearchCV
grid_search_rf = GridSearchCV(pipeline, param_grid, cv=3, scoring=scorer)

# Ajustar el GridSearchCV utilizando los datos de entrenamiento y las etiquetas
grid_search_rf.fit(X_mixto_train, y_mixto_train)


# In[111]:


print(evaluate_model(grid_search_rf,X_mixto_test, y_mixto_test))


# ### Neuronal network

# In[112]:


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', MLPClassifier())
])

param_grid = {
    'preprocess__text__ngram_range': [(1, 1), (1, 2)],  # Ejemplo de parámetro para TfidfVectorizer
    'model__hidden_layer_sizes': [(50,), (100,)]  # Ejemplo de parámetro para LogisticRegression
}

# Crear el objeto GridSearchCV
grid_search_rn = GridSearchCV(pipeline, param_grid, cv=3, scoring=scorer)

# Ajustar el GridSearchCV utilizando los datos de entrenamiento y las etiquetas
grid_search_rn.fit(X_mixto_train, y_mixto_train)


# In[113]:


print(evaluate_model(grid_search_rn,X_mixto_test, y_mixto_test))


# ### Remark 
# 
# In the mixed phase, the Random Forest model showed outstanding performance in triage classification.
# 
# Those metrics reflect a significant improvement compared to previous phases, demonstrating the effectiveness of the model in accurately classifying triage categories. Specifically, the model achieved high accuracy in category 1, with an f1-score of 0.61, and also performed well in categories 2, 3, and 4, with f1-scores of 0.67, 0.61, and 0.59, respectively.

# ## Results

# 1. In the numerical phase, the best model was a neural network. Although this model was able to adequately classify some categories, it is observed that the general metrics are moderate, indicating the need for improvements.
# 
# |Class| Logistic Regression    | Random Forest       | Neural Network  |
# |------|-----------|-----------|-----------|
# |1| 0.26| 0.40   | 0.40   |
# |2| 0.46| 0.47   | 0.50   |
# |3| 0.40| 0.43   | 0.38   |
# |4|0.38|0.42|0.46|
# |**F1 macro Score**|0.40|0.43|0.44| 
# 
# 2. In the text phase, logistic regression stood out as the best model, showing better performance compared to the numerical phase.
#     These metrics show a significant improvement in classification ability, especially in terms of accuracy and f1-score.

# |Class| Logistic Regression    | Random Forest       | Neural Network  |
# |------|-----------|-----------|-----------|
# |1| 0.67| 0.56   | 0.55   |
# |2| 0.64| 0.62   | 0.60   |
# |3| 0.60| 0.57   | 0.55   |
# |4|0.58|0.56|0.57|
# |**F1 macro Score**|0.61|0.59|0.56| 
# 
# 3. In the mixed phase, the Random Forest model showed outstanding performance in triage classification.

# |Class| Logistic Regression    | Random Forest       | Neural Network  |
# |------|-----------|-----------|-----------|
# |1| 0.61| 0.61   | 0.54   |
# |2| 0.65| 0.67   | 0.62   |
# |3| 0.60| 0.61   | 0.57   |
# |4|0.60|0.59|0.59|
# |**F1 macro Score**|0.61|0.62|0.58| 
# 
# These metrics reflect a significant improvement compared to previous phases, demonstrating the effectiveness of the model in accurately classifying triage categories. Specifically, the model achieved high accuracy in category 1, with an f1-score of 0.61, and also performed well in categories 2, 3, and 4, with f1-scores of 0.67, 0.61, and 0.59, respectively.

# In[114]:


grid_search_rf.best_params_


# In[115]:


# Accede a los resultados del Grid Search
resultados = grid_search_rf.cv_results_

# Obtén las métricas obtenidas en cada paso del Grid Search
mejores_metricas = resultados['mean_test_score']
std_metricas = resultados['std_test_score']
configuraciones = resultados['params']

# Itera sobre las configuraciones y sus métricas correspondientes
for metrica, std, config in zip(mejores_metricas, std_metricas, configuraciones):
    print("Configuración:", config)
    print("Métrica media:", metrica)
    print("Desviación estándar:", std)
    print("------------------------")


# | Configuration                                     | Mean F1 score | Standard deviation |
# |--------------------------------------------------|---------------|---------------------|
# | 'model__n_estimators': 50, 'preprocess__text__ngram_range': (1, 1) | 0.58        | 0.008              |
# | 'model__n_estimators': 50, 'preprocess__text__ngram_range': (1, 2) | 0.56        | 0.003              |
# | 'model__n_estimators': 100, 'preprocess__text__ngram_range': (1, 1)| 0.59        | 0.006              |
# | 'model__n_estimators': 100, 'preprocess__text__ngram_range': (1, 2)| 0.57        | 0.007              |
# 

# The third evaluated configuration, with {'model__n_estimators': 100, 'preprocess__text__ngram_range': (1, 1)}, stands out as the best model choice. It achieved a higher average metric score of 0.590, indicating superior performance compared to other configurations. Furthermore, the low standard deviation of 0.0063 demonstrates consistent results across iterations, showcasing the model's stability and reliability. This configuration strikes a balance between complexity and simplicity, utilizing 100 estimators and considering only individual words. Together, these factors contribute to the model's strong and consistent performance.

# ## Save the final model
# 
# Taking into account that we have already identified the model with the best performance under the parameters defined at the beginning. The code is ready to save the model in pkl format and be used with an interface for its deployment. However, that is not part of the proposed scope of this project.

# In[116]:


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


# In[117]:


save_model(grid_search_rf, 'final_model.pkl' )


# ## Demo:
# 
# A set of 10 records is taken at random and the model is tested to work correctly.

# In[118]:


X_mixto_test.sample(10, random_state = 123)


# In[119]:


grid_search_rf.predict(X_mixto_test.sample(10, random_state = 123))


# ## Conclusions
# In summary, this project addressed the challenge of triage classification in health services. Using various models and techniques, we developed a comprehensive solution that supports healthcare professionals to accurately triage patients.
# 
# Regarding improvements: It is suggested to continue the investigation to improve the experiment. It is important to highlight that the developed model is ready to be complemented with an application that facilitates its deployment and achieves a comprehensive end-to-end solution.
# 
# This complementary tool aims to support and improve the work of doctors in emergency situations, allowing them to make more accurate and faster decisions in triage classification.

# In[ ]:




