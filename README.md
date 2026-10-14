# Actividad 5 - Design Thinking: Definiendo e Ideando

Este repositorio contiene la implementación end-to-end de un flujo de Machine Learning con preparación de datos, entrenamiento con validación cruzada y trazabilidad de experimentos mediante **MLflow**.

---

## 📁 Estructura del Repositorio

```text
Actividad5/
├── datos/
│   ├── datos_ini/          # Dataset original extraído
│   └── datos_limp/         # Datasets limpios, deduplicados y divididos (train/test)
├── fuentes/
│   ├── datos_prep.py       # Script de limpieza, imputación y deduplicación
│   ├── train.py            # Script de entrenamiento, CV y registros en MLflow
│   └── entrena.ipynb       # Notebook ejecutable en Google Colab
├── evidencias/
│   └── evidencia_mlflow.png # Gráfica de comparación de experimentos
├── README.md               # Guía de reproducción y documentación
└── CHANGELOG.md            # Registro de cambios y resultados finales
```

---

## 🛠️ Guía de Reproducción

1. **Instalar dependencias:**
   ```bash
   pip install mlflow scikit-learn pandas matplotlib
   ```

2. **Ejecutar la preparación de datos:**
   ```bash
   python fuentes/datos_prep.py
   ```

3. **Ejecutar el entrenamiento y registro en MLflow:**
   ```bash
   python fuentes/train.py
   ```

---

## 📊 Evidencias de MLflow y Métricas Comparativas

Se compararon dos algoritmos con validación cruzada (5-fold):

| Algoritmo | Accuracy | F1-Score | Cross-Validation F1 (Mean) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 1.0000 | 1.0000 | 0.9583 |
| **Logistic Regression** | 1.0000 | 1.0000 | 0.9667 |

### Evidencia de Registro
La evidencia gráfica del registro de experimentos se encuentra en la carpeta `evidencias/evidencia_mlflow.png`.
