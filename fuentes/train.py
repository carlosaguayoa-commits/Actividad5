
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configurar MLflow
os.makedirs("mlruns_db", exist_ok=True)
os.makedirs("evidencias", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns_db/mlflow.db")
mlflow.set_experiment("Actividad_5_MLflow")

# Cargar datos limpios
train_df = pd.read_csv("datos/datos_limp/train.csv")
test_df = pd.read_csv("datos/datos_limp/test.csv")

X_train = train_df.drop(columns=['especie'])
y_train = train_df['especie']
X_test = test_df.drop(columns=['especie'])
y_test = test_df['especie']

modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42)
}

resultados = []

# Declarar tipos de confianza para modelos basados en árboles en skops
tipos_confianza = ["sklearn.tree._tree.Tree"]

for nombre, model in modelos.items():
    with mlflow.start_run(run_name=f"Run_{nombre}"):
        # Validación cruzada 5-fold
        scores_cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        cv_mean = scores_cv.mean()
        
        # Entrenamiento
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        
        # Logs en MLflow
        mlflow.log_param("algoritmo", nombre)
        mlflow.log_metric("cv_f1_mean", cv_mean)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Registrar el modelo pasando skops_trusted_types para evitar el UntrustedTypesFoundException
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo",
            skops_trusted_types=tipos_confianza
        )
        
        resultados.append({"Algoritmo": nombre, "Accuracy": acc, "F1-Score": f1, "CV_F1": cv_mean})

# Generar evidencia gráfica de resultados registrados
res_df = pd.DataFrame(resultados)
plt.figure(figsize=(8, 4))
plt.bar(res_df['Algoritmo'], res_df['F1-Score'], color=['#2ca02c', '#1f77b4'])
plt.title("Evidencia de Experimentos en MLflow - F1-Score")
plt.ylabel("Score")
plt.ylim(0, 1.1)
for i, v in enumerate(res_df['F1-Score']):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
plt.savefig("evidencias/evidencia_mlflow.png")
plt.close()

print("Experimentos completados y evidencias guardadas en /evidencias sin errores.")
