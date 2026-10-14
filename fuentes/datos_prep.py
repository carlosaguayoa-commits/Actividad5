
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def cargar_y_limpiar_datos():
    os.makedirs("datos/datos_ini", exist_ok=True)
    os.makedirs("datos/datos_limp", exist_ok=True)
    
    # CARGA DATASET IRIS
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.rename(columns={'target': 'especie'}, inplace=True)
    
    # GUARDAR DATASET
    df.to_csv("datos/datos_ini/dataset_original.csv", index=False)
    
    # LIMPIEZA DE DATOS
    df_limpio = df.drop_duplicates()
    df_limpio = df_limpio.fillna(df_limpio.median(numeric_only=True))
    
    # GUARDAR DATASET LIMPIO
    df_limpio.to_csv("datos/datos_limp/dataset_limpio.csv", index=False)
    
    # DIVIDIR DATOS DE ENTRENAMIENTO Y PRUEBA
    train, test = train_test_split(df_limpio, test_size=0.2, random_state=42)
    train.to_csv("datos/datos_limp/train.csv", index=False)
    test.to_csv("datos/datos_limp/test.csv", index=False)
  

if __name__ == "__main__":
    cargar_y_limpiar_datos()
