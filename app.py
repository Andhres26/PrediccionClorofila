from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model

import joblib

app = Flask(__name__)

# Cargar los modelos y el scaler
model_a = load_model('modelos/modelo_clorofila_a_aguacate.h5')
model_b = load_model('modelos/modelo_clorofila_b_aguacate.h5')
scaler = joblib.load('modelos/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener los valores del formulario
        gari = float(request.form['gari'])
        mnli = float(request.form['mnli'])
        gndvi = float(request.form['gndvi'])

        # Crear un DataFrame con los valores
        nueva_muestra = pd.DataFrame({'GARI': [gari], 'MNLI': [mnli], 'GNDVI': [gndvi]})

        # Normalizar la muestra
        nueva_muestra_normalizada = scaler.transform(nueva_muestra)

        # Hacer las predicciones
        clorofila_a_predicha = model_a.predict(nueva_muestra_normalizada)[0][0]
        clorofila_b_predicha = model_b.predict(nueva_muestra_normalizada)[0][0]

        # Mostrar los resultados en la página
        return render_template('index.html', 
                               gari=gari, 
                               mnli=mnli, 
                               gndvi=gndvi, 
                               clorofila_a=clorofila_a_predicha, 
                               clorofila_b=clorofila_b_predicha)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)