import pandas as pd

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

"""


# Separación de los datos y etiquetas para el dataset de Diabetes
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']

# Separación de los datos y etiquetas para el dataset quirúrgico
X_surgical = surgical_data.drop('complication', axis=1)
y_surgical = surgical_data['complication']

# División del dataset de Diabetes en entrenamiento, validación y prueba
X_train_d, X_temp_d, y_train_d, y_temp_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)
X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(X_temp_d, y_temp_d, test_size=0.5, random_state=42)

# División del dataset quirúrgico en entrenamiento, validación y prueba
X_train_s, X_temp_s, y_train_s, y_temp_s = train_test_split(X_surgical, y_surgical, test_size=0.3, random_state=42)
X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(X_temp_s, y_temp_s, test_size=0.5, random_state=42)

# Normalización de los datos
scaler = StandardScaler()

# Normalizar el dataset de Diabetes
X_train_d = scaler.fit_transform(X_train_d)
X_val_d = scaler.transform(X_val_d)
X_test_d = scaler.transform(X_test_d)

# Normalizar el dataset quirúrgico
X_train_s = scaler.fit_transform(X_train_s)
X_val_s = scaler.transform(X_val_s)
X_test_s = scaler.transform(X_test_s)

# Entrenamiento de Regresión Logística para el dataset de Diabetes
logreg_diabetes = LogisticRegression()
logreg_diabetes.fit(X_train_d, y_train_d)

# Entrenamiento y evaluación de Regresión Logística y KNN en ambos datasets

# 1. Regresión Logística para Diabetes
logreg_diabetes = LogisticRegression()
logreg_diabetes.fit(X_train_d, y_train_d)
y_pred_val_d_logreg = logreg_diabetes.predict(X_val_d)

# Evaluación de Regresión Logística para Diabetes
print('Resultados de Regresión Logística para Diabetes:')
print('Accuracy:', accuracy_score(y_val_d, y_pred_val_d_logreg))
print('Precision:', precision_score(y_val_d, y_pred_val_d_logreg))
print('Recall:', recall_score(y_val_d, y_pred_val_d_logreg))
print('Confusion Matrix:\n', confusion_matrix(y_val_d, y_pred_val_d_logreg))

# 2. KNN para Diabetes
knn_diabetes = KNeighborsClassifier(n_neighbors=5)
knn_diabetes.fit(X_train_d, y_train_d)
y_pred_val_d_knn = knn_diabetes.predict(X_val_d)

# Evaluación de KNN para Diabetes
print('Resultados de KNN para Diabetes:')
print('Accuracy:', accuracy_score(y_val_d, y_pred_val_d_knn))
print('Precision:', precision_score(y_val_d, y_pred_val_d_knn))
print('Recall:', recall_score(y_val_d, y_pred_val_d_knn))
print('Confusion Matrix:\n', confusion_matrix(y_val_d, y_pred_val_d_knn))

# 3. Regresión Logística para el dataset quirúrgico
logreg_surgical = LogisticRegression()
logreg_surgical.fit(X_train_s, y_train_s)
y_pred_val_s_logreg = logreg_surgical.predict(X_val_s)

# Evaluación de Regresión Logística para Quirúrgico
print('Resultados de Regresión Logística para Quirúrgico:')
print('Accuracy:', accuracy_score(y_val_s, y_pred_val_s_logreg))
print('Precision:', precision_score(y_val_s, y_pred_val_s_logreg))
print('Recall:', recall_score(y_val_s, y_pred_val_s_logreg))
print('Confusion Matrix:\n', confusion_matrix(y_val_s, y_pred_val_s_logreg))

# 4. KNN para el dataset quirúrgico
knn_surgical = KNeighborsClassifier(n_neighbors=5)
knn_surgical.fit(X_train_s, y_train_s)
y_pred_val_s_knn = knn_surgical.predict(X_val_s)

# Evaluación de KNN para Quirúrgico
print('Resultados de KNN para Quirúrgico:')
print('Accuracy:', accuracy_score(y_val_s, y_pred_val_s_knn))
print('Precision:', precision_score(y_val_s, y_pred_val_s_knn))
print('Recall:', recall_score(y_val_s, y_pred_val_s_knn))
print('Confusion Matrix:\n', confusion_matrix(y_val_s, y_pred_val_s_knn))