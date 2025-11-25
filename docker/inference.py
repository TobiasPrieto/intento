import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime

def cargar_artefactos():
    print("Cargando modelos y configuraciones...")
    config = joblib.load('imputation_config.joblib')
    kmeans = joblib.load('kmeans.joblib')
    encoder = joblib.load('encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
    
    # Cargar coordenadas
    try:
        coords_df = pd.read_csv('coordenadas_aus.csv')
        print("Coordenadas cargadas correctamente.")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar coordenadas_aus.csv: {e}")
        coords_df = None
        
    return config, kmeans, encoder, scaler, model, coords_df

def preprocesar_datos(df, config, kmeans, encoder, scaler, coords_df=None):
    # 0. Merge con Coordenadas si están disponibles y no están en el input
    if coords_df is not None and 'Latitud' not in df.columns and 'Location' in df.columns:
        print("Buscando coordenadas para las locaciones...")
        # Merge left para mantener las filas del input
        df = df.merge(coords_df, on="Location", how="left")
        
        # Verificar si quedaron nulos (ciudades no encontradas)
        missing_coords = df[df['Latitud'].isnull()]['Location'].unique()
        if len(missing_coords) > 0:
            print(f"Advertencia: No se encontraron coordenadas para: {missing_coords}")
            # Podríamos asignar un default o dejar que falle/impute más adelante si es crítico
            
    # 1. Fechas
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    # 2. Clustering (Regiones)
    # Aseguramos que Latitud y Longitud existan (aunque sean NaN)
    if 'Latitud' not in df.columns: df['Latitud'] = np.nan
    if 'Longitud' not in df.columns: df['Longitud'] = np.nan
    
    coords = df[['Latitud', 'Longitud']]
    
    # Si tenemos coordenadas válidas, predecimos cluster. Si no, manejamos el error o default.
    # Para simplificar, si hay NaNs en coords, kmeans.predict fallará. 
    # Llenamos con 0 o algún valor por defecto si es necesario, o asumimos que el merge funcionó.
    # Aquí asumiremos que si falló el merge, el usuario debería haber provisto coords o aceptamos el error.
    
    # Hack para evitar error si hay NaNs en coords (ej. ciudad desconocida)
    # En producción idealmente se valida antes.
    coords_filled = coords.fillna(0) 
    
    if 'NorfolkIsland' in df['Location'].values:
         df.loc[df['Location'] == 'NorfolkIsland', 'RegionCluster'] = 5
    
    # Solo predecimos para los que no son NorfolkIsland (o sobreescribimos, el orden importa)
    # Nota: kmeans.predict espera array sin nans
    try:
        df['RegionCluster'] = kmeans.predict(coords_filled)
    except:
        df['RegionCluster'] = 0 # Fallback
        
    if 'NorfolkIsland' in df['Location'].values:
         df.loc[df['Location'] == 'NorfolkIsland', 'RegionCluster'] = 5


    # 3. Imputaciones (Versión simplificada usando medianas/modas globales guardadas)
    # Rellenar Numéricos
    for col, median_val in config['medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
            
    # Rellenar Categóricos
    for col, mode_val in config['modes'].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
            
    # RainToday logic
    if 'RainToday' not in df.columns:
        if 'Rainfall' in df.columns:
            df.loc[df['Rainfall'] > 0, 'RainToday'] = 'Yes'
            df.loc[df['Rainfall'] == 0, 'RainToday'] = 'No'
        else:
            df['RainToday'] = 'No'

    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0)

    # 4. Feature Engineering (Vientos)
    wind_map = config['wind_dir_map']
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if col in df.columns:
            df[col] = df[col].map(wind_map)
            df[f'{col}_sin'] = np.sin(np.deg2rad(df[col]))
            df[f'{col}_cos'] = np.cos(np.deg2rad(df[col]))

    # 5. Feature Engineering (Diferencias)
    # Asegurar que existan las columnas para restar, si no, crear con NaN (se imputarán o fallarán si son críticas)
    # El modelo espera estas columnas. Si faltan en el input, deberían haber sido imputadas o estar presentes.
    # Asumimos que el input tiene las columnas base o que la imputación de arriba las cubrió si estaban en config['medians']
    
    # Definir pares para diferencias
    pairs = [
        ('Temp9am', 'Temp3pm', 'temp_diff'),
        ('Humidity9am', 'Humidity3pm', 'humidity_diff'),
        ('WindSpeed9am', 'WindSpeed3pm', 'wind_diff'),
        ('MinTemp', 'MaxTemp', 'min_max_diff'),
        ('Cloud9am', 'Cloud3pm', 'cloud_diff'),
        ('Pressure9am', 'Pressure3pm', 'pressure_diff')
    ]
    
    for c1, c2, diff_col in pairs:
        if c1 in df.columns and c2 in df.columns:
            df[diff_col] = abs(df[c1] - df[c2])
        else:
            df[diff_col] = 0 # o NaN

    # 6. Encoding (OHE)
    cols_cat = ['Month', 'RegionCluster']
    encoded_array = encoder.transform(df[cols_cat])
    encoded_cols = encoder.get_feature_names_out(cols_cat)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=cols_cat), df_encoded], axis=1)

    # 7. Selección y Escalado
    # Asegurar que las columnas coincidan con el entrenamiento
    features_escalar = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm','temp_diff',
        'humidity_diff', 'wind_diff', 'min_max_diff', 'cloud_diff', 'pressure_diff']
    
    # Verificar que todas las features estén
    for f in features_escalar:
        if f not in df.columns:
            df[f] = 0 # Imputar con 0 si falta algo crítico post-ingeniería
    
    df[features_escalar] = scaler.transform(df[features_escalar])
    
    # Eliminar columnas que no entran al modelo
    cols_drop = ['Date','Location','WindGustDir','WindDir9am','WindDir3pm','Latitud','Longitud']
    # Filtrar solo las que existan
    cols_drop = [c for c in cols_drop if c in df.columns]
    X = df.drop(columns=cols_drop)
    
    # Reordenar columnas para coincidir con el modelo (importante en sklearn)
    # Si el modelo tiene feature_names_in_, usarlos
    if hasattr(model, 'feature_names_in_'):
        X = X[model.feature_names_in_]
    
    return X

if __name__ == "__main__":
    # 1. Cargar recursos
    config, kmeans, encoder, scaler, model, coords_df = cargar_artefactos()

    # 2. Cargar datos de entrada desde CSV
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'input.csv'
    print(f"Leyendo datos de entrada desde: {input_file}")
    
    try:
        df_input = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error al leer el archivo de entrada {input_file}: {e}")
        sys.exit(1)
    
    try:
        # 3. Procesar
        # Nota: preprocesar_datos maneja el dataframe completo
        X = preprocesar_datos(df_input, config, kmeans, encoder, scaler, coords_df)
        
        # 4. Predecir
        predictions = model.predict(X)
        probas = model.predict_proba(X)
        
        print(f"\n{'='*30}")
        print("RESULTADOS DE LA INFERENCIA")
        print(f"{'='*30}")
        
        for i, (pred, prob) in enumerate(zip(predictions, probas)):
            res = "LLOVERÁ" if pred == 1 else "NO LLOVERÁ"
            # Intentar mostrar fecha/locación si existen en el original para contexto
            loc = df_input.iloc[i].get('Location', 'N/A')
            date = df_input.iloc[i].get('Date', 'N/A')
            
            print(f"Fila {i+1} ({loc} - {date}):")
            print(f"  Pronóstico: {res}")
            print(f"  Probabilidad de lluvia: {prob[1]:.2%}")
            print("-" * 20)
            
        print(f"{'='*30}\n")
        
    except Exception as e:
        print(f"Error en inferencia: {e}")
        import traceback
        traceback.print_exc()