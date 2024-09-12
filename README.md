# IA Proyecto 1

Este proyecto tiene como objetivo aplicar diversas técnicas de clasificación de datos en dos conjuntos de datos: uno relacionado con la predicción de diabetes y otro con la predicción de enfermedades cardíacas. El propósito es explorar herramientas relacionadas con Machine Learning y contribuir al desarrollo del conocimiento mediante la investigación y comparación de modelos.

## Estructura del Proyecto

- `diabetes.csv`: Conjunto de datos del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales (NIDDK) para predecir la presencia de diabetes.
- `heart_disease_uci.csv`: Conjunto de datos de complicaciones quirúrgicas, que será utilizado para predecir el riesgo de complicaciones postoperatorias basadas en variables clínicas.

## Modelos Utilizados

El proyecto implementa los siguientes algoritmos de Machine Learning:

- **Regresión Logística**: Usado para la clasificación binaria y multiclase.
- **K-Nearest Neighbors (KNN)**: Método basado en la proximidad de los datos para realizar clasificaciones.

## Requisitos

- Python 3.7+
- Jupyter Notebook

### Paquetes de Python

Para ejecutar este proyecto, es necesario instalar los siguientes paquetes de Python. Puedes encontrarlos en el archivo `requirements.txt`:

- `pandas`: Para la manipulación de datos.
- `matplotlib`: Para la creación de gráficos y visualización de datos.
- `seaborn`: Biblioteca de visualización estadística.
- `scikit-learn`: Para la implementación de algoritmos de Machine Learning.

Puedes instalar estos paquetes ejecutando:

```bash
pip install -r requirements.txt