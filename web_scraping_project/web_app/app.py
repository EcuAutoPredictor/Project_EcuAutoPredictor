from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import json
from collections import defaultdict

app = Flask(__name__)

# Cargar el modelo entrenado y los encoders guardados
#modelo_rf = joblib.load('../notebooks/models/modelo_random_forest.pkl')
modelo_rf = joblib.load('../notebooks/models/modelo_xgboost.pkl')
with open('../notebooks/encoders/encoders_xgboost.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Definir las columnas categóricas y el orden de las features
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']

# Preparar las opciones para los combo boxes a partir de los encoders (las clases conocidas)
dropdown_options = {}
for col in categorical_cols:
    dropdown_options[col] = list(encoders[col].classes_)

# Generar un mapeo de Marca a Modelos usando el dataset (se asume que 'archivo_unido_FINAL3.csv' contiene estas columnas)
df_data = pd.read_csv('../notebooks/data/archivo_unido_FINAL3.csv')
marca_modelo_map = defaultdict(list)
for index, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
    marca = row['Marca']
    modelo = row['Modelo']
    marca_modelo_map.setdefault(marca, []).append(modelo)
# Convertir el mapeo a JSON para enviarlo al frontend
marca_modelo_json = json.dumps(dict(marca_modelo_map))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}
    if request.method == "POST":
        # Recoger los datos del formulario
        data = {
            'Marca': request.form.get('Marca'),
            'Modelo': request.form.get('Modelo'),
            'Provincia': request.form.get('Provincia'),
            'Año': request.form.get('Año'),
            'Kilometraje': request.form.get('Kilometraje'),
            'Transmisión': request.form.get('Transmisión'),
            'Dirección': request.form.get('Dirección'),
            'Motor': request.form.get('Motor'),
            'Tracción': request.form.get('Tracción'),
            'Color': request.form.get('Color'),
            'Combustible': request.form.get('Combustible')
        }
        form_data = data

        # Crear un DataFrame con los datos ingresados
        df_input = pd.DataFrame([data], columns=features)
        # Convertir campos numéricos
        df_input['Año'] = pd.to_numeric(df_input['Año'], errors='coerce')
        df_input['Kilometraje'] = pd.to_numeric(df_input['Kilometraje'], errors='coerce')
        df_input['Motor'] = pd.to_numeric(df_input['Motor'], errors='coerce')
        
        # Aplicar los encoders a las columnas categóricas
        for col in categorical_cols:
            # Si el valor ingresado no está en las clases conocidas, se asigna el primer valor
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]
            df_input[col] = encoders[col].transform(df_input[col].astype(str))
        
        df_input = df_input[features]
        prediction = modelo_rf.predict(df_input)[0]
    
    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

if __name__ == '__main__':
    app.run(debug=True)
