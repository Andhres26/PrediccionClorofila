from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import joblib

app = Flask(__name__)

# Lista de cultivos y sus modelos
cultivos = {
    'aguacate': ('modelos/modelo_clorofila_a_aguacate.h5', 'modelos/modelo_clorofila_b_aguacate.h5'),
    'maiz': ('modelos/modelo_clorofila_a_maiz.h5', 'modelos/modelo_clorofila_b_maiz.h5'),
    'cafe': ('modelos/modelo_clorofila_a_cafe.h5', 'modelos/modelo_clorofila_b_cafe.h5')
}

# Cargar el scaler
scaler = joblib.load('modelos/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    clorofila_a_predicha, clorofila_b_predicha = None, None
    gari, mnli, gndvi, cultivo_seleccionado = None, None, None, None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            cultivo_seleccionado = request.form['cultivo']
            gari = float(request.form['gari'])
            mnli = float(request.form['mnli'])
            gndvi = float(request.form['gndvi'])

            # Verificar si el cultivo seleccionado tiene modelos
            if cultivo_seleccionado in cultivos:
                modelo_a_path, modelo_b_path = cultivos[cultivo_seleccionado]
                model_a = load_model(modelo_a_path)
                model_b = load_model(modelo_b_path)
            else:
                return render_template('index.html', error="Cultivo no válido", cultivos=cultivos.keys())

            # Crear DataFrame y normalizar
            nueva_muestra = pd.DataFrame({'GARI': [gari], 'MNLI': [mnli], 'GNDVI': [gndvi]})
            nueva_muestra_normalizada = scaler.transform(nueva_muestra)

            # Hacer predicciones
            clorofila_a_predicha = model_a.predict(nueva_muestra_normalizada)[0][0]
            clorofila_b_predicha = model_b.predict(nueva_muestra_normalizada)[0][0]
        
        except ValueError:
            return render_template('index.html', error="Por favor ingrese valores numéricos válidos", cultivos=cultivos.keys())

    return render_template('index.html', 
                           gari=gari, mnli=mnli, gndvi=gndvi, 
                           clorofila_a=clorofila_a_predicha, 
                           clorofila_b=clorofila_b_predicha, 
                           cultivo_seleccionado=cultivo_seleccionado,
                           cultivos=cultivos.keys())

if __name__ == '__main__':
    app.run(debug=True)
