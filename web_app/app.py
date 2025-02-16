from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 📌 Rutas de los archivos de modelos (ajusta si es necesario)
MODEL_PATH = "../models/random_forest.pkl"  # O "../models/xgboost.pkl"
ENCODERS_PATH = "../models/encoders/"

# 📌 Cargar el modelo entrenado
model = joblib.load(MODEL_PATH)

# 📌 Cargar encoders para variables categóricas
label_columns = ["modelo", "marca", "pais", "clase", "sub_clase", "tipo", "tipo_combustible"]
label_encoders = {col: joblib.load(f"{ENCODERS_PATH}/label_encoder_{col}.pkl") for col in label_columns}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 📌 Capturar datos del formulario
        nuevo_auto = {
            "modelo": request.form.get("modelo"),
            "marca": request.form.get("marca"),
            "pais": request.form.get("pais"),
            "year_modelo": float(request.form.get("year_modelo")),
            "clase": request.form.get("clase"),
            "sub_clase": request.form.get("sub_clase"),
            "tipo": request.form.get("tipo"),
            "cilindraje": float(request.form.get("cilindraje")),
            "tipo_combustible": request.form.get("tipo_combustible")
        }
        
        # 📌 Convertir valores categóricos con Label Encoders
        for col in label_columns:
            if nuevo_auto[col] in label_encoders[col].classes_:
                nuevo_auto[col] = label_encoders[col].transform([nuevo_auto[col]])[0]
            else:
                nuevo_auto[col] = -1  # Si no existe, asignamos -1

        # 📌 Convertir a DataFrame y reordenar columnas según el modelo
        df_nuevo_auto = pd.DataFrame([nuevo_auto])
        df_nuevo_auto = df_nuevo_auto[model.feature_names_in_]
        
        # 📌 Hacer la predicción
        predicted_price = model.predict(df_nuevo_auto)[0]
        
        return render_template('result.html', predicted_price=predicted_price)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
