# Breast Cancer Prediction - Logistic Regression


Este proyecto implementa un modelo de regresión logística para predecir si un tumor de mama es maligno o benigno, utilizando el dataset "Breast Cancer Wisconsin (Diagnostic)" de Kaggle. El análisis incluye exploración de datos, selección de características, manejo de desbalance de clases y evaluación del modelo.

## Descripción del Proyecto

El notebook `Breast_Cancer_Prediction_Logistic_Regression.ipynb` realiza un análisis completo de datos:
1. **Carga y limpieza de datos**  
   - Eliminación de valores nulos.  
   - Exclusión de columnas irrelevantes.  

2. **Análisis exploratorio de datos (EDA)**  
   - Visualización de distribuciones y correlaciones.  
   - Detección de valores atípicos.  

3. **Preprocesamiento**  
   - Selección de características para evitar multicolinealidad.  
   - Balanceo de clases mediante SMOTE.  
   - Escalado de variables numéricas.  

4. **Entrenamiento del modelo**  
   - Implementación de regresión logística.  
   - Optimización de hiperparámetros con `GridSearchCV`.  

5. **Evaluación del modelo**  
   - Cálculo de métricas como precisión, AUC-ROC y matriz de confusión.  

El modelo alcanzó una **precisión del 95%**, demostrando un desempeño sólido en la clasificación de tumores.

## Fuentes de Datos

El dataset utilizado es [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) de Kaggle, que contiene 569 muestras con 30 características numéricas derivadas de imágenes digitalizadas de aspiración con aguja fina (FNA) de masas mamarias.

## Estructura del Proyecto

- **`.gitignore`**: Archivo para especificar archivos y carpetas que Git debe ignorar (ej. datos temporales, entornos virtuales, archivos de IDE).
- **`environment.yml`**: Archivo de configuración para crear un entorno Conda con todas las dependencias necesarias (Python, bibliotecas como Pandas, scikit-learn, etc.).
- **`README.md`**: Este archivo, con descripción del proyecto, instrucciones de instalación y uso.
- **`data/`**: Carpeta que contiene el dataset utilizado.
  - `data_cancer.csv`: Dataset (descárgalo de Kaggle y colócalo aquí).
- **`Notebooks/`**: Carpeta con el notebook de análisis.
  - `Breast_Cancer_Prediction_Logistic_Regression.ipynb`: Notebook principal con el análisis y modelo.


## Requisitos e Instalación

### Cómo ejecutar
1. Clona el repositorio:
   ```bash
    git clone https://github.com/AlejandroSegura24/Breast-Cancer-Prediction-Logistic-Regression.git
    cd Breast-Cancer-Prediction-Logistic-Regression
   ```

2. Crea un entorno virtual con Conda (recomendado):

    ```bash
    conda env create -f environment.yml
    conda activate cancer_prediction
    ```

3. Descarga `data_cancer.csv` de Kaggle y colócalo en la raíz del proyecto.

## Tecnologías y habilidades demostradas

- **Lenguaje:** Python.  
- **Bibliotecas principales:**
  - `pandas`: Limpieza, exploración y manipulación de datos.
  - `numpy`: Operaciones numéricas y manejo de matrices.
  - `matplotlib`: Visualización de métricas y distribuciones de datos.
  - `scikit-learn`: Implementación del modelo de regresión logística, escalado de datos, SMOTE y evaluación de métricas.
- **Habilidades:**
  - Limpieza y preparación de datos (detección de valores atípicos y eliminación de variables redundantes).
  - Análisis exploratorio de datos (EDA) con visualizaciones informativas.
  - Preprocesamiento de datos: escalado, balanceo de clases y selección de características.
  - Entrenamiento y validación de modelos supervisados.
  - Evaluación de desempeño mediante métricas estadísticas.
  - Documentación clara del flujo de trabajo y resultados dentro del notebook.

## Autor
- **Nombre**: David Alejandro Segura 
- **Contacto**: davidalejandrocmbs@gmail.com
- **Propósito:** Proyecto educativo enfocado en aplicar técnicas de análisis de datos y aprendizaje automático con Python para la detección temprana del cáncer de mama mediante regresión logística.