import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar los datos de diabetes
diabetes_data = pd.read_csv('diabetes.csv')

# Cargar el segundo conjunto de datos
surgical_data = pd.read_csv('Surgical-deepnet.csv')


# Ver las primeras filas de cada dataset
# print(diabetes_data.head())
# print(surgical_data.head())

# Resumen estadístico
print(diabetes_data.describe())
print(surgical_data.describe())


# Comprobar si hay valores faltantes
# print(diabetes_data.isnull().sum())
# print(surgical_data.isnull().sum())


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

# Cargar el segundo conjunto de datos modificado
surgical_data_mod = pd.read_csv('Surgical-deepnet_modificado.csv')

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

# ---- Para el dataset de Diabetes ----

# Separar características (X) y variable objetivo (y)
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']

# Generar todas las combinaciones posibles de las columnas de X (Diabetes)
features_diabetes = X_diabetes.columns
combinations_diabetes = []
for r in range(1, len(features_diabetes)+1):
    combinations_diabetes.extend(itertools.combinations(features_diabetes, r))

# Dividir el dataset en 70% entrenamiento y 30% temporal
X_train_d, X_temp_d, y_train_d, y_temp_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)

# Dividir el conjunto temporal en 15% validación y 15% prueba
X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(X_temp_d, y_temp_d, test_size=0.5, random_state=42)



# ---- Para el dataset quirúrgico ----

# Separar características (X) y variable objetivo (y)
X_surgical = surgical_data_mod.drop('complication', axis=1)
y_surgical = surgical_data_mod['complication']

# Generar todas las combinaciones posibles de las columnas de X (Surgical)
features_surgical = X_surgical.columns
combinations_surgical = []
for r in range(1, len(features_surgical)+1):
    combinations_surgical.extend(itertools.combinations(features_surgical, r))

# Dividir el dataset quirúrgico en 70% entrenamiento y 30% temporal
X_train_s, X_temp_s, y_train_s, y_temp_s = train_test_split(X_surgical, y_surgical, test_size=0.3, random_state=42)

# Dividir el conjunto temporal en 15% validación y 15% prueba
X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(X_temp_s, y_temp_s, test_size=0.5, random_state=42)

# Lista para almacenar combinaciones y sus métricas
combinations_with_metrics = []

# Iterar sobre cada combinación de características (Diabetes)
for combo in combinations_diabetes:
    # Seleccionar las características correspondientes para entrenamiento y validación
    X_train_combo = X_train_d[list(combo)]
    X_val_combo = X_val_d[list(combo)]
    
    # Entrenar un modelo de Regresión Logística
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train_combo, y_train_d)
    
    # Validar el modelo usando el conjunto de validación
    y_pred_val = logreg.predict(X_val_combo)
    
    # Calcular precisión, precision, recall, f1-score y matriz de confusión
    accuracy_d = accuracy_score(y_val_d, y_pred_val)
    precision_d = precision_score(y_val_d, y_pred_val, zero_division=1)
    recall_d = recall_score(y_val_d, y_pred_val)
    f1_d = f1_score(y_val_d, y_pred_val)
    confusion_d = confusion_matrix(y_val_d, y_pred_val)
    
    # Almacenar las combinaciones y sus métricas
    combinations_with_metrics.append((combo, accuracy_d, precision_d, recall_d, f1_d, confusion_d))

# Ordenar las combinaciones por F1-Score en orden descendente
combinations_with_metrics.sort(key=lambda x: x[4], reverse=True)

# Mostrar las 5 mejores combinaciones
print("Top 5 combinaciones de características (ordenadas por F1-Score):")
for combo, accuracy, precision, recall, f1, confusion in combinations_with_metrics[:5]:
    print(f"Combinación: {combo}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Matriz de Confusión en Validación:\n{confusion}\n")

# Lista para almacenar combinaciones y sus métricas
combinations_with_metrics = []

# Iterar sobre cada combinación de características (Diabetes)
for combo in combinations_diabetes:
    # Seleccionar las características correspondientes para entrenamiento y validación
    X_train_combo = X_train_d[list(combo)]
    X_val_combo = X_val_d[list(combo)]
    
    # Entrenar un modelo de Regresión Logística
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train_combo, y_train_d)
    
    # Validar el modelo usando el conjunto de validación
    y_pred_val = logreg.predict(X_val_combo)
    
    # Calcular precisión, precision, recall, f1-score y matriz de confusión
    accuracy_d = accuracy_score(y_val_d, y_pred_val)
    precision_d = precision_score(y_val_d, y_pred_val, zero_division=1)
    recall_d = recall_score(y_val_d, y_pred_val)
    f1_d = f1_score(y_val_d, y_pred_val)
    confusion_d = confusion_matrix(y_val_d, y_pred_val)
    
    # Almacenar las combinaciones y sus métricas
    combinations_with_metrics.append((combo, accuracy_d, precision_d, recall_d, f1_d, confusion_d))

# Ordenar las combinaciones por F1-Score en orden descendente
combinations_with_metrics.sort(key=lambda x: x[4], reverse=True)

# Mostrar las 5 mejores combinaciones
print("Top 5 combinaciones de características (ordenadas por F1-Score):")
for combo, accuracy, precision, recall, f1, confusion in combinations_with_metrics[:5]:
    print(f"Combinación: {combo}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Matriz de Confusión en Validación:\n{confusion}\n")


# Cargar el dataset de Diabetes
diabetes_data_knn = pd.read_csv('diabetes.csv')

# Separar características (X) y variable objetivo (y)
X_diabetes_knn = diabetes_data_knn.drop('Outcome', axis=1)
y_diabetes_knn = diabetes_data_knn['Outcome']

# Generar todas las combinaciones posibles de las columnas de X (Diabetes)
features_diabetes_knn = X_diabetes_knn.columns
combinations_diabetes_knn = []
for r in range(1, len(features_diabetes_knn)+1):
    combinations_diabetes_knn.extend(itertools.combinations(features_diabetes_knn, r))

# Dividir el dataset de Diabetes en 70% entrenamiento y 30% restante (que luego se dividirá en validación y prueba)
X_train_d_knn, X_temp_d_knn, y_train_d_knn, y_temp_d_knn = train_test_split(X_diabetes_knn, y_diabetes_knn, test_size=0.3, random_state=42)

# Dividir el 30% restante en 15% para validación y 15% para testing
X_val_d_knn, X_test_d_knn, y_val_d_knn, y_test_d_knn = train_test_split(X_temp_d_knn, y_temp_d_knn, test_size=0.5, random_state=42)

# Lista para almacenar combinaciones y sus métricas
combinations_with_metrics_knn_diabetes = []

# Iterar sobre cada combinación de características (Diabetes)
for combo in combinations_diabetes_knn:
    # Seleccionar las características correspondientes para entrenamiento y validación
    X_train_combo_d_knn = X_train_d_knn[list(combo)]
    X_val_combo_d_knn = X_val_d_knn[list(combo)]
    X_test_combo_d_knn = X_test_d_knn[list(combo)]
    
    # Entrenar un modelo de KNN con el conjunto de entrenamiento
    knn_diabetes = KNeighborsClassifier(n_neighbors=5)
    knn_diabetes.fit(X_train_combo_d_knn, y_train_d_knn)
    
    # Validar el modelo usando el conjunto de validación
    y_pred_val_d_knn = knn_diabetes.predict(X_val_combo_d_knn)
    
    # Evaluar el modelo en el conjunto de validación
    accuracy_d_knn = accuracy_score(y_val_d_knn, y_pred_val_d_knn)
    precision_d_knn = precision_score(y_val_d_knn, y_pred_val_d_knn, zero_division=1)
    recall_d_knn = recall_score(y_val_d_knn, y_pred_val_d_knn)
    f1_d_knn = f1_score(y_val_d_knn, y_pred_val_d_knn)
    confusion_d_knn = confusion_matrix(y_val_d_knn, y_pred_val_d_knn)
    
    # Almacenar las combinaciones y sus métricas
    combinations_with_metrics_knn_diabetes.append((combo, accuracy_d_knn, precision_d_knn, recall_d_knn, f1_d_knn, confusion_d_knn))

# Ordenar las combinaciones por F1-Score en orden descendente
combinations_with_metrics_knn_diabetes.sort(key=lambda x: x[4], reverse=True)

# Mostrar el Top 5 combinaciones
print("Top 5 combinaciones de características de KNN Diabetes (ordenadas por F1-Score):")
for combo, accuracy, precision, recall, f1, confusion in combinations_with_metrics_knn_diabetes[:5]:
    print(f"Combinación: {combo}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Matriz de Confusión en Validación:\n{confusion}\n")

    # Cargar el dataset quirúrgico modificado
surgical_data_knn = pd.read_csv('Surgical-deepnet_modificado.csv')

# Separar características (X) y variable objetivo (y)
X_surgical_knn = surgical_data_knn.drop('complication', axis=1)
y_surgical_knn = surgical_data_knn['complication']

# Generar todas las combinaciones posibles de las columnas de X (Surgical)
features_surgical_knn = X_surgical_knn.columns
combinations_surgical_knn = []
for r in range(1, len(features_surgical_knn)+1):
    combinations_surgical_knn.extend(itertools.combinations(features_surgical_knn, r))

# Dividir el dataset quirúrgico en 70% entrenamiento y 30% restante (que luego se dividirá en validación y prueba)
X_train_s_knn, X_temp_s_knn, y_train_s_knn, y_temp_s_knn = train_test_split(X_surgical_knn, y_surgical_knn, test_size=0.3, random_state=42)

# Dividir el 30% restante en 15% para validación y 15% para testing
X_val_s_knn, X_test_s_knn, y_val_s_knn, y_test_s_knn = train_test_split(X_temp_s_knn, y_temp_s_knn, test_size=0.5, random_state=42)

# Lista para almacenar combinaciones y sus métricas
combinations_with_metrics_knn_surgical = []

# Iterar sobre cada combinación de características (Surgical)
for combo in combinations_surgical_knn:
    # Seleccionar las características correspondientes para entrenamiento y validación
    X_train_combo_s_knn = X_train_s_knn[list(combo)]
    X_val_combo_s_knn = X_val_s_knn[list(combo)]
    X_test_combo_s_knn = X_test_s_knn[list(combo)]
    
    # Entrenar un modelo de KNN con el conjunto de entrenamiento
    knn_surgical = KNeighborsClassifier(n_neighbors=5)
    knn_surgical.fit(X_train_combo_s_knn, y_train_s_knn)
    
    # Validar el modelo usando el conjunto de validación
    y_pred_val_s_knn = knn_surgical.predict(X_val_combo_s_knn)
    
    # Evaluar el modelo en el conjunto de validación
    accuracy_s_knn = accuracy_score(y_val_s_knn, y_pred_val_s_knn)
    precision_s_knn = precision_score(y_val_s_knn, y_pred_val_s_knn, zero_division=1)
    recall_s_knn = recall_score(y_val_s_knn, y_pred_val_s_knn)
    f1_s_knn = f1_score(y_val_s_knn, y_pred_val_s_knn)
    confusion_s_knn = confusion_matrix(y_val_s_knn, y_pred_val_s_knn)
    
    # Almacenar las combinaciones y sus métricas
    combinations_with_metrics_knn_surgical.append((combo, accuracy_s_knn, precision_s_knn, recall_s_knn, f1_s_knn, confusion_s_knn))

# Ordenar las combinaciones por F1-Score en orden descendente
combinations_with_metrics_knn_surgical.sort(key=lambda x: x[4], reverse=True)

# Mostrar el Top 5 combinaciones
print("Top 5 combinaciones de características de KNN Quirúrgico (ordenadas por F1-Score):")
for combo, accuracy, precision, recall, f1, confusion in combinations_with_metrics_knn_surgical[:5]:
    print(f"Combinación: {combo}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Matriz de Confusión en Validación:\n{confusion}\n")