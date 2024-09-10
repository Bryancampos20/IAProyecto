import pandas as pd

# Cargar los datos de diabetes
diabetes_data = pd.read_csv('diabetes.csv')

# Cargar el segundo conjunto de datos
surgical_deepnet_data = pd.read_csv('Surgical-deepnet.csv')

"""
# Ver las primeras filas de cada dataset
print(diabetes_data.head())
print(surgical_deepnet_data.head())
"""

"""
# Resumen estadístico
print(diabetes_data.describe())
print(surgical_deepnet_data.describe())

"""

# Comprobar si hay valores faltantes
print(diabetes_data.isnull().sum())
print(surgical_deepnet_data.isnull().sum())