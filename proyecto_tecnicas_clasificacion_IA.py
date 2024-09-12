import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


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

# Visualización de la distribución de clases para el dataset de Diabetes
sns.countplot(x='Outcome', data=diabetes_data)
plt.title('Distribución de Clases en el Dataset de Diabetes')
plt.show()

# Visualización de la distribución de la variable de complicaciones
sns.countplot(x='complication', data=surgical_data)
plt.title('Distribución de Clases en el Dataset Quirúrgico (Complication)')
plt.show()




# Obtener las columnas excepto la columna 'Outcome'
features = diabetes_data.columns.drop('Outcome')

# Crear una figura con múltiples subplots
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):

    # Mostrar el resumen estadístico para cada feature
    print(f"Descripción de {feature}:")
    print(diabetes_data[feature].describe())
    print("\n")

    # Crear el histograma
    plt.subplot(3, 3, i)  # Configurar el layout en 3 filas y 3 columnas
    plt.hist(diabetes_data[diabetes_data['Outcome'] == 0][feature], color='blue', alpha=0.5, label='No Diabetes', bins=20)
    plt.hist(diabetes_data[diabetes_data['Outcome'] == 1][feature], color='red', alpha=0.5, label='Diabetes', bins=20)
    plt.title(f'Histograma de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.show()

# Obtener las columnas excepto la columna 'complication'
features = surgical_data.columns.drop('complication')

# Crear una figura con múltiples subplots
plt.figure(figsize=(18, 15))

# Ajustamos el layout a 6x4 para acomodar las 21 características
for i, feature in enumerate(features, 1):
    
    # Mostrar el resumen estadístico para cada feature
    print(f"Descripción de {feature}:")
    print(surgical_data[feature].describe())
    print("\n")

    # Crear el histograma
    plt.subplot(6, 4, i)  # Configurar el layout en 6 filas y 4 columnas
    plt.hist(surgical_data[surgical_data['complication'] == 0][feature], color='blue', alpha=0.5, label='No Complication', bins=20)
    plt.hist(surgical_data[surgical_data['complication'] == 1][feature], color='red', alpha=0.5, label='Complication', bins=20)
    plt.title(f'Histograma de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.show()

# Crear el pairplot para todas las características
sns.pairplot(diabetes_data, hue='Outcome', palette="coolwarm")

# Mostrar el gráfico
plt.show()


# Crear el pairplot para todas las características, coloreando por 'complication'
sns.pairplot(surgical_data, hue='complication', palette="coolwarm")

# Mostrar el gráfico
plt.show()

"""

# Cargar el segundo conjunto de datos modificado
surgical_data_mod = pd.read_csv('Surgical-deepnet_modificado.csv')
"""
# Obtener las columnas excepto la columna 'complication'
features = surgical_data_mod.columns.drop('complication')

# Crear una figura con múltiples subplots
plt.figure(figsize=(18, 15))

# Ajustamos el layout a 6x4 para acomodar las 10 características
for i, feature in enumerate(features, 1):
    
    # Mostrar el resumen estadístico para cada feature
    print(f"Descripción de {feature}:")
    print(surgical_data_mod[feature].describe())
    print("\n")

    # Crear el histograma
    plt.subplot(6, 4, i)  # Configurar el layout en 6 filas y 4 columnas
    plt.hist(surgical_data_mod[surgical_data_mod['complication'] == 0][feature], color='blue', alpha=0.5, label='No Complication', bins=20)
    plt.hist(surgical_data_mod[surgical_data_mod['complication'] == 1][feature], color='red', alpha=0.5, label='Complication', bins=20)
    plt.title(f'Histograma de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.show()

# Crear el pairplot para todas las características
sns.pairplot(surgical_data_mod, hue='complication', palette="coolwarm")

# Mostrar el gráfico
plt.show()
"""
# Cargar el dataset quirúrgico modificado
surgical_data_mod = pd.read_csv('Surgical-deepnet_modificado.csv')

# ---- Para el dataset de Diabetes ----
# Separar características (X) y variable objetivo (y)
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']

# Generar todas las combinaciones posibles de las columnas de X (Diabetes)
features_diabetes = X_diabetes.columns
combinations_diabetes = []
for r in range(1, len(features_diabetes)+1):
    combinations_diabetes.extend(itertools.combinations(features_diabetes, r))

# Dividir el dataset de Diabetes en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)

# Lista para almacenar combinaciones y su precisión si es mayor a 0.8
combinations_with_high_accuracy = []

# Iterar sobre cada combinación de características (Diabetes)
for combo in combinations_diabetes:
    # Seleccionar las características correspondientes
    X_train_combo_d = X_train_d[list(combo)]
    X_test_combo_d = X_test_d[list(combo)]
    
    # Entrenar un modelo de Regresión Logística
    logreg_diabetes = LogisticRegression(max_iter=10000)
    logreg_diabetes.fit(X_train_combo_d, y_train_d)
    
    # Predecir en el conjunto de prueba
    y_pred_d = logreg_diabetes.predict(X_test_combo_d)
    
    # Calcular la precisión
    accuracy_d = accuracy_score(y_test_d, y_pred_d)
    
    # Mostrar la precisión de todas las combinaciones
    # print(f"Precisión del modelo de Diabetes con las características {combo}: {accuracy_d}\n")
    
    # Si la precisión es mayor a 0.8, almacenarla en la lista
    if accuracy_d > 0.76:
        combinations_with_high_accuracy.append((combo, accuracy_d))

# Mostrar las combinaciones con precisión mayor a 0.8
print("Combinaciones de características con precisión mayor a 0.8:")
for combo, accuracy in combinations_with_high_accuracy:
    print(f"Combinación: {combo}, Precisión: {accuracy}")

# ---- Para el dataset quirúrgico ----
# Separar características (X) y variable objetivo (y)
X_surgical = surgical_data_mod.drop('complication', axis=1)
y_surgical = surgical_data_mod['complication']

# Generar todas las combinaciones posibles de las columnas de X (Surgical)
features_surgical = X_surgical.columns
combinations_surgical = []
for r in range(1, len(features_surgical)+1):
    combinations_surgical.extend(itertools.combinations(features_surgical, r))

# Dividir el dataset quirúrgico en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_surgical, y_surgical, test_size=0.3, random_state=42)

# Lista para almacenar combinaciones y su precisión si es mayor a 0.8
combinations_with_high_accuracy_surgical = []

# Iterar sobre cada combinación de características (Surgical)
for combo in combinations_surgical:
    # Seleccionar las características correspondientes
    X_train_combo_s = X_train_s[list(combo)]
    X_test_combo_s = X_test_s[list(combo)]
    
    # Entrenar un modelo de Regresión Logística
    logreg_surgical = LogisticRegression(max_iter=10000)
    logreg_surgical.fit(X_train_combo_s, y_train_s)
    
    # Predecir en el conjunto de prueba
    y_pred_s = logreg_surgical.predict(X_test_combo_s)
    
    # Calcular la precisión
    accuracy_s = accuracy_score(y_test_s, y_pred_s)
    
    # Mostrar la precisión de todas las combinaciones
    # print(f"Precisión del modelo Quirúrgico con las características {combo}: {accuracy_s}\n")
    
    # Si la precisión es mayor a 0.8, almacenarla en la lista
    if accuracy_s > 0.8:
        combinations_with_high_accuracy_surgical.append((combo, accuracy_s))

# Mostrar las combinaciones con precisión mayor a 0.8
print("Combinaciones de características quirúrgicas con precisión mayor a 0.8:")
for combo, accuracy in combinations_with_high_accuracy_surgical:
    print(f"Combinación: {combo}, Precisión: {accuracy}")

# ---- Visualización de las características para el dataset de Diabetes ----
# Obtener las columnas excepto la columna 'Outcome'
features = diabetes_data.columns.drop('Outcome')

# Crear una figura con múltiples subplots
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.hist(diabetes_data[diabetes_data['Outcome'] == 0][feature], color='blue', alpha=0.5, label='No Diabetes', bins=20)
    plt.hist(diabetes_data[diabetes_data['Outcome'] == 1][feature], color='red', alpha=0.5, label='Diabetes', bins=20)
    plt.title(f'Histograma de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.show()

# ---- Visualización de las características para el dataset quirúrgico ----
# Obtener las columnas excepto la columna 'complication'
features = surgical_data_mod.columns.drop('complication')

# Crear una figura con múltiples subplots
plt.figure(figsize=(18, 15))

for i, feature in enumerate(features, 1):
    plt.subplot(6, 4, i)
    plt.hist(surgical_data_mod[surgical_data_mod['complication'] == 0][feature], color='blue', alpha=0.5, label='No Complication', bins=20)
    plt.hist(surgical_data_mod[surgical_data_mod['complication'] == 1][feature], color='red', alpha=0.5, label='Complication', bins=20)
    plt.title(f'Histograma de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.show()

# Crear el pairplot para todas las características en el dataset quirúrgico
sns.pairplot(surgical_data_mod, hue='complication', palette="coolwarm")
plt.show()