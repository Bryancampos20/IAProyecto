import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos de diabetes
diabetes_data = pd.read_csv('diabetes.csv')

# Cargar el segundo conjunto de datos
surgical_data = pd.read_csv('Surgical-deepnet.csv')

"""
# Ver las primeras filas de cada dataset
print(diabetes_data.head())
print(surgical_data.head())

# Resumen estadístico
print(diabetes_data.describe())
print(surgical_data.describe())


# Comprobar si hay valores faltantes
print(diabetes_data.isnull().sum())
print(surgical_data.isnull().sum())


# Revisar las estadísticas descriptivas
print(diabetes_data.describe())
print(surgical_data.describe())

"""

# Visualización de la distribución de clases para el dataset de Diabetes
sns.countplot(x='Outcome', data=diabetes_data)
plt.title('Distribución de Clases en el Dataset de Diabetes')
plt.show()

# Visualización de la distribución de clases para el dataset quirúrgico
sns.countplot(x='Outcome', data=surgical_data)
plt.title('Distribución de Clases en el Dataset Quirúrgico')
plt.show()
