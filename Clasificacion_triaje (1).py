#!/usr/bin/env python
# coding: utf-8

# # Definición del proyecto
# 
# 
# ### **Motivación**
# En Colombia, al igual que en muchos otros países, garantizar la correcta clasificación de los pacientes que acuden a los servicios de urgencias de las instituciones de salud representa un desafío crucial. La adecuada clasificación de triaje es fundamental para asegurar que cada paciente reciba la atención adecuada en el momento preciso, sobre todo en situaciones críticas donde cada minuto cuenta.
# 
# En este contexto, es una necesidad ayudar a  mejorar el proceso de clasificación de triaje en Colombia. Esto implica la implementación de herramientas y enfoques que permitan a los profesionales de la salud tomar decisiones más precisas y rápidas al clasificar a los pacientes.
# 
# Teniendo en cuenta la situación actual, la contribución principal de este trabajo es el desarrollo de una herramienta que ayude a los profesionales de la salud en la clasificación de triaje.
# 
# ### **Problema**
# 
# Desarrollar un modelo de clasificación que permita clasificar a los pacientes que llegan a una sala de urgencias, teniendo en cuenta caracteristicas como su temperatura, oximetria, edad entre otras en conjunto con el texto registrado por el profesional de la salud al  momento de atender a los pacientes.
# 
# ### **Métricas y evaluación**
# 
# Teniendo en cuenta que se busca clasificar a los pacientes en 5 categorias de triage, estamos frente a un problema de clasificación multiclase. Como las clases no estan balanceadas, la métrica que se usara para evaluar el desempeño de los modelos será **f1 macro score** pues esta pondera la precisión y la exhaustividad.
# 
# 

# ## Requerimientos:

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


# ## Descripción de los datos
# 
# Para el desarrollo de este proyecto se tiene un conjunto de $20.000$ registros de pacientes que han asistido a urgencias entre el año 2021 y 2022. La información esta divida en $2$ conjuntos de datos:
# 
# 1. **base_datos**: Base de datos de clientes con las siguientes variables:
# 
# 
# *   ID_TM_Afiliado_cod: Id único de cada paciente.
# *   Edad: Edad del paciente.
# *   Genero: Genero del paciente.
# *   SedeAtencion: Luegar de atención.
# *   TipoAtencion: Tipo de atención recibida.
# *   TipoAfiliado: Tipo de afiliación del paciente.    
# 
# 
# 2. **base_consulta**: Base de datos de las consultas con las siguientes variables:
# 
# *    ID_TM_Afiliado_cod: Id único de cada paciente.
# *    IdConsulta_cod: ID único de la consulta.
# *    Fecha: Fecha de la consulta.
# *    Dia: Día de la consulta.
# *    Mes: Mes de la consulta.
# *    AÑO: Año de la consulta.
# *    TAS: Tensión arterial sistólica.
# *    TAD: Tensión arterial diastólica.
# *    TAM: Tensión arterial media.
# *    FC: Frecuencia cardiaca.
# *    FR: Frecuencia respiratoria.
# *    Temp: Temperatura.
# *    Oximetria: Oximetria.
# *    Anotacion TRIAGE Consulta: Texto registrado por el especialista.
# *   ClasificaTriage: Triaje asignado al paciente

# In[6]:


url_base_datos = 'base_datos.csv'
url_base_consulta = 'base_consulta.csv'


# In[7]:


base_datos = pd.read_csv(url_base_datos).drop('Unnamed: 0', axis = 1)
base_consulta = pd.read_csv(url_base_consulta).drop('Unnamed: 0', axis = 1)


# ## Analisis Exploratorio
# 
# ### Base de pacientes

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
    Crea un gráfico comparativo de una variable numérica utilizando un boxplot y un histograma.

    Argumentos:
    - var1: variable numérica para crear el gráfico comparativo.
    - x_label1: etiqueta del eje x del boxplot y histograma.
    - y_label1: etiqueta del eje y del boxplot.
    - title1: título del boxplot.
    - title2: título del histograma.

    Retorna:
    - None: muestra el gráfico comparativo.
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


# In[138]:


base_datos.Edad.describe()


# In[15]:


comp_grap(base_datos.Edad, 'Edad', 'Edad en años', 'Box plot Edad', 'Distribución de la edad')


# In[16]:


def bar_graps(var, x_label, y_label, title, rot):
    """
    Crea un gráfico de barras a partir de una variable categórica.

    Argumentos:
    - var: variable categórica para crear el gráfico de barras.
    - x_label: etiqueta del eje x.
    - y_label: etiqueta del eje y.
    - title: título del gráfico.
    - rot: ángulo de rotación de las etiquetas del eje x (en grados).

    Retorna:
    - None: muestra el gráfico de barras.
    """
    plt.bar(var.value_counts().index.to_list(), var.value_counts().to_list() )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=rot)
    plt.show()


# In[17]:


# Distribución de lugar de atención
bar_graps(base_datos.SedeAtencion, 'Sede de atención', 'Valores', 'Gráfico de barras de sede de atención', 45)


# In[18]:


# Distribución de lugar de atención
bar_graps(base_datos.Genero, 'Genero', 'Valores', 'Gráfico de barras de Genero', 0)


# In[137]:


base_datos.Genero.value_counts(normalize = True)


# In[19]:


# Distribución de lugar de atención
bar_graps(base_datos.TipoAfiliado, 'Tipo afiliación', 'Valores', 'Gráfico de barras de Tipo afiliación', 0)


# ### Base de consultas

# In[20]:


base_consulta.head(2)


# In[21]:


# Nro de registros y columnas
base_consulta.shape


# In[22]:


# Nro duplicados
base_consulta.duplicated().sum()


# In[23]:


# Nro valores unicos por columna
base_consulta.nunique()


# In[24]:


# Valores faltantes
base_consulta.isna().sum()


# In[25]:


# Distribución de lugar de atención
bar_graps(base_consulta.ClasificaTriage , 'Etiqueta Triage', 'Valores', 'Gráfico de barras de Clasificación Triage', 0)


# In[143]:


base_consulta.ClasificaTriage.value_counts(normalize = True)


# In[26]:


base_consulta['TAM'] = base_consulta['TAM'].str.replace(',', '.').astype(float)


# In[27]:


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


# In[28]:


comp_grap(base_consulta.TAM, 'TAM', 'Escala TAM', 'Box plot TAM', 'Distribución de TAM')


# In[29]:


comp_grap(base_consulta.FC, 'FC', 'Escala FC', 'Box plot FC', 'Distribución de FC')


# In[30]:


comp_grap(base_consulta.FR, 'FR', 'Escala FR', 'Box plot FR', 'Distribución de FR')


# In[31]:


comp_grap(base_consulta.Oximetria, 'Oximetria', 'Escala Oximetria', 'Box plot Oximetria', 'Distribución de Oximetria')


# In[32]:


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


# In[142]:


base_consulta.groupby('AÑO')['AÑO'].count()/20000


# In[141]:


base_consulta.groupby('DIA')['DIA'].count().sort_values(ascending = False)/20000


# In[140]:


base_consulta.groupby('MES')['MES'].count().sort_values(ascending = False)/20000


# ## Desarrollo fase de modelación
# 
# En este notebook se presenta un análisis completo del entrenamiento de cuatro algoritmos para resolver un problema multiclase. La primera fase consiste en la carga y preparación de datos numéricos y de texto, que serán utilizados en el entrenamiento de los modelos. Se detallan los procesos de limpieza, normalización y selección de variables relevantes para cada algoritmo.
# 
# En la segunda fase, se lleva a cabo el entrenamiento de los algoritmos utilizando únicamente los datos numéricos, utilizando la metodología de grid search para la búsqueda de los mejores hiperparámetros y su correspondiente evaluación. Se presentan las métricas obtenidas para cada modelo.
# 
# La tercera fase repite el proceso anterior, pero incluyendo los datos de texto. Se muestra cómo los modelos mejoran al incorporar esta información adicional y se comparan los resultados con la fase anterior.
# 
# En la cuarta fase, se realiza un entrenamiento conjunto con los datos numéricos y de texto. Se utiliza nuevamente la metodología de grid search y se evalúan los modelos resultantes. Además, se extraen las características más importantes y se presentan los resultados relevantes del modelo seleccionado.
# 
# En resumen, este notebook brinda una visión completa del proceso de entrenamiento de varios algoritmos para resolver un problema multiclase, utilizando diferentes metodologías y datos. Se presentan las métricas y resultados obtenidos en cada fase, lo que permite comparar los diferentes modelos y determinar cuál es el más adecuado para el problema en cuestión.
# 
# 
# 
# 

# # Tratamiento de los datos

# In[33]:


base = base_datos.merge(base_consulta, on = ['ID_TM_Afiliado_cod'], how = 'left').drop_duplicates(['ID_TM_Afiliado_cod', 'IdConsulta_cod'])


# In[34]:


base.nunique()


# In[35]:


base.head(2)


# In[36]:


base['ClasificaTriage'].value_counts(normalize = True )


# In[37]:


# Filtro clase diferente a 0
base = base.query('ClasificaTriage > 0')


# In[38]:


base.isna().sum()


# In[39]:


base['Temp']= base['Temp'].fillna(base['Temp'].mean())
#base = base.dropna()


# In[40]:


base['ClasificaTriage'].value_counts( )


# # Selección de variables
# ## Limpieza

# In[41]:


def limpieza(text):
  """
    Realiza la limpieza de un texto eliminando caracteres no alfanuméricos y convirtiéndolo a minúsculas.

    Argumentos:
    - text: texto a limpiar.

    Retorna:
    - text_limpio: texto limpio sin caracteres no alfanuméricos y en minúsculas.
    """
  text_limpio = re.sub('[\W]+', ' ', text.lower())
  return text_limpio


# In[42]:


base['final'] = base['Anotacion TRIAGE Consulta'].apply(limpieza).apply(str)


# ## Determinación del número de palabras

# In[43]:


my_stopwords = stopwords.words('spanish')


# In[44]:


tfidf = TfidfVectorizer(stop_words= my_stopwords)


# In[45]:


docs = base['final'].values


# In[46]:


X = tfidf.fit_transform(docs)


# In[47]:


X.shape


# In[48]:


variance = []
for i in [100, 200, 300, 400,500, 600, 700, 800, 900,1000]:
    tfidf_temp = TfidfVectorizer(stop_words= my_stopwords, max_features=i)
    X_temp = tfidf_temp.fit_transform(docs)
    variance.append(X_temp.sum()/X.sum())
    print(i, X_temp.sum()/X.sum())


# In[49]:


plt.xlabel('Número de palabras')
plt.ylabel('Varianza explicada')
plt.plot([100, 200, 300, 400,500, 600, 700, 800, 900,1000], variance)
plt.axvline(x=800, linestyle='--', color='r') # Línea vertical en 500
plt.hlines(y=0.90, xmin=0, xmax=1000, linestyle='--', color='r') # Línea horizontal en 0.95
plt.show()


# ## Preparación de los datos de texto

# In[50]:


# Base solo texto
X_texto = base['final'].values
y_texto= base['ClasificaTriage'].values


# In[51]:


my_stopwords = stopwords.words('spanish')


# In[52]:


vectorizer_texto = TfidfVectorizer(stop_words= my_stopwords)
tfidf_matrix = vectorizer_texto.fit_transform(X_texto)


# In[53]:


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


# In[54]:


important_words_index = get_most_important_words_index(tfidf_matrix, vectorizer_texto, n=800)


# In[55]:


# Selecciona las columnas correspondientes a las palabras más importantes
X_texto_reduced = tfidf_matrix[:, important_words_index]
X_texto_reduced.shape


# In[56]:


#claves_coincidentes = [str(clave) + '-' + str(valor) for clave, valor in vectorizer_texto.vocabulary_.items() if valor in important_words_index]


# In[57]:


#palabra_ind = pd.DataFrame(claves_coincidentes)

#palabra_ind[['palabra', 'indice']] = palabra_ind[0].str.split('-', n=1, expand=True)

#palabra_ind = palabra_ind[['palabra', 'indice']].reset_index()

#palabra_ind['index'] = palabra_ind['index'] + 5


# In[58]:


#pd.DataFrame(important_words_index).to_csv('indice_palabras_v1.csv')


# In[59]:


# Obtenemos los nombres de las palabras
palabras = vectorizer_texto.vocabulary_.keys()

# Calculamos el puntaje TF-IDF promedio para cada palabra
puntajes = tfidf_matrix.mean(axis=0).tolist()[0]

# Creamos una lista de tuplas (palabra, puntaje) y ordenamos por puntaje
palabras_puntajes = list(zip(palabras, puntajes))
palabras_puntajes.sort(key=lambda x: x[1], reverse=True)

print("Las 10 palabras más importantes:")
for palabra, puntaje in palabras_puntajes[:10]:
    print("- {} ({:.4f})".format(palabra, puntaje))


# In[60]:


#palabras = pd.DataFrame(palabras_puntajes)


# In[61]:


#palabras['palabras'] = palabras[0]
#palabras['puntaje'] = palabras[1]


# In[62]:


#palabras[['palabras', 'puntaje']].to_csv('ranking.csv')


# ## Variables numéricas y categoricas importantes

# In[63]:


base_numerica = base[['Edad', 'Genero',  'ClasificaTriage', 'SedeAtencion' , 'TipoAfiliado',  'TAM','FC', 'FR', 'Temp', 'Oximetria']]


# In[64]:


X_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)
y_num= base_numerica['ClasificaTriage'].values


# In[65]:


#k_best = SelectKBest(f_classif, k=6)
#X_new = k_best.fit_transform(X_num, y_num)
#X_num = X_num.iloc[:,list(k_best.get_support())]


# In[66]:


scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)


# In[67]:


X_num


# In[68]:


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


# In[69]:


var_random_forest = importancia_variables_modelo(X_num, y_num, RandomForestClassifier())


# In[70]:


var_logistic_reg = importancia_variables_modelo(X_num, y_num, LogisticRegression())


# In[71]:


var_red = importancia_variables_modelo(X_num, y_num, MLPClassifier())


# In[72]:


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


# ## Preparación de los datos numéricos

# In[73]:


X_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)

k_best = SelectKBest(f_classif, k=6)
X_new = k_best.fit_transform(X_num, y_num)
X_num = X_num.iloc[:,list(k_best.get_support())]


# In[74]:


X_num.shape


# In[75]:


#X_num = X_num[:,k_best.get_support()]


# In[76]:


mask = k_best.get_support()


# In[77]:


base_num = pd.get_dummies(base_numerica, drop_first = True).drop('ClasificaTriage', axis = 1)


# In[78]:


fvalues, pvalues = f_classif(base_num.iloc[:, mask], y_num)


# In[79]:


df = pd.DataFrame({'Característica': base_num.columns[mask], 'F': fvalues, 'p-valor': pvalues})
df.sort_values('F')


# # Modelación numérica
# ## Definición de la métrica a optimizar

# In[80]:


def evaluate_model(model,x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)

    clsReport = classification_report( y_test,y_pred )
    return clsReport


# In[81]:


X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_scaled, y_num, test_size=0.2, random_state=42)


# In[82]:


scorer = make_scorer(f1_score, average='macro')


# ## Regresión Logistica

# In[83]:


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


# In[84]:


print(evaluate_model(cv_reg,X_num_test, y_num_test))


# ## Random Forest

# In[85]:


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


# In[86]:


print(evaluate_model(cv_rf,X_num_test, y_num_test))


# ## Red Neuronal

# In[87]:


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


# In[88]:


print(evaluate_model(cv_rn,X_num_test, y_num_test))


# # Modelación de texto
# ## Regresión logistica
# 

# In[89]:


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


# In[90]:


X_texto = base['Anotacion TRIAGE Consulta'].values
y_texto = base['ClasificaTriage'].values


# In[91]:


X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_texto, y_texto, test_size=0.2, random_state=42)


# In[92]:


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


# In[93]:


print(evaluate_model(cv_reg_text,X_text_test, y_text_test))


# ## Random Forest

# In[94]:


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


# In[95]:


print(evaluate_model(cv_rf_text,X_text_test, y_text_test))


# ## Red Neuronal

# In[96]:


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


# In[97]:


print(evaluate_model(cv_rn_text,X_text_test, y_text_test))


# # Mixto

# ## Prepareación de los datos

# In[98]:


def split_dataframe(df1):
    df1 = pd.get_dummies(df1, columns = ['Genero', 'SedeAtencion', 'TipoAfiliado'],  drop_first = True)
    df1 = df1[['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante', 'Anotacion TRIAGE Consulta', 'ClasificaTriage']]
    #X_texto = df1['Anotacion TRIAGE Consulta'].values
    return df1


# In[99]:


split_dataframe(base)


# In[100]:


# Desde aqui
X = split_dataframe(base)[['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante', 'Anotacion TRIAGE Consulta']]
y = split_dataframe(base)['ClasificaTriage']
X_mixto_train, X_mixto_test, y_mixto_train, y_mixto_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[101]:


# Crear el ColumnTransformer para encadenar los transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), ['TAM', 'FC', 'FR', 'Genero_Masculino', 'SedeAtencion_VS UUBC LAS AMERICAS', 'TipoAfiliado_Cotizante']),
        ('text', TfidfVectorizer(tokenizer=tokenize, stop_words = my_stopwords, max_features= 800), 'Anotacion TRIAGE Consulta')
    ])


# ### Regrsión lineal

# In[102]:


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


# In[103]:


print(evaluate_model(grid_search_lr,X_mixto_test, y_mixto_test))


# ### Random Forest

# In[104]:


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


# In[105]:


print(evaluate_model(grid_search_rf,X_mixto_test, y_mixto_test))


# ### Red Neuronal

# In[106]:


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


# In[107]:


print(evaluate_model(grid_search_rn,X_mixto_test, y_mixto_test))


# In[108]:


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


# In[109]:


save_model(grid_search_rf, 'final_model.pkl' )


# In[135]:


grid_search_rf.predict(X_mixto_test.sample(10, random_state = 123))


# In[136]:


X_mixto_test.sample(10, random_state = 123)


# In[ ]:




