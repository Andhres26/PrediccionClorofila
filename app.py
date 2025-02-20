from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Lista de cultivos y sus modelos
cultivos = {
    'Aguacate': ('modelos/modelo_clorofila_a_aguacate.h5', 'modelos/modelo_clorofila_b_aguacate.h5'),
    'Maiz': ('modelos/modelo_clorofila_a_maiz.h5', 'modelos/modelo_clorofila_b_maiz.h5'),
    'Café': ('modelos/modelo_clorofila_a_cafe2.h5', 'modelos/modelo_clorofila_b_cafe2.h5')
}

# Cargar el scaler
scaler_path = 'modelos/scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError(f"No se encontró el archivo de normalización: {scaler_path}")

# Cargar modelos una sola vez (evita cargarlos en cada request)
modelos_cargados = {}
for cultivo, (modelo_a, modelo_b) in cultivos.items():
    if os.path.exists(modelo_a) and os.path.exists(modelo_b):
        modelos_cargados[cultivo] = (load_model(modelo_a), load_model(modelo_b))
    else:
        raise FileNotFoundError(f"Modelos de {cultivo} no encontrados en {modelo_a} o {modelo_b}")

@app.route('/', methods=['GET', 'POST'])
def index():
    clorofila_a_predicha, clorofila_b_predicha = None, None
    gari, mnli, gndvi, cultivo_seleccionado = None, None, None, None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            cultivo_seleccionado = request.form.get('cultivo')
            gari = float(request.form.get('gari'))
            mnli = float(request.form.get('mnli'))
            gndvi = float(request.form.get('gndvi'))

            # Validar el cultivo
            if cultivo_seleccionado not in modelos_cargados:
                return render_template('index.html', error="Cultivo no válido", cultivos=cultivos.keys())

            # Obtener modelos precargados
            model_a, model_b = modelos_cargados[cultivo_seleccionado]

            # Crear DataFrame y normalizar
            nueva_muestra = pd.DataFrame({'GARI': [gari], 'MNLI': [mnli], 'GNDVI': [gndvi]})
            nueva_muestra_normalizada = scaler.transform(nueva_muestra)

            # Hacer predicciones
            clorofila_a_predicha = float(model_a.predict(nueva_muestra_normalizada)[0][0])
            clorofila_b_predicha = float(model_b.predict(nueva_muestra_normalizada)[0][0])

        except ValueError:
            return render_template('index.html', error="Por favor ingrese valores numéricos válidos", cultivos=cultivos.keys())
        except Exception as e:
            return render_template('index.html', error=f"Error en la predicción: {str(e)}", cultivos=cultivos.keys())

    return render_template('index.html', 
                           gari=gari, mnli=mnli, gndvi=gndvi, 
                           clorofila_a=clorofila_a_predicha, 
                           clorofila_b=clorofila_b_predicha, 
                           cultivo_seleccionado=cultivo_seleccionado,
                           cultivos=cultivos.keys())

if __name__ == '__main__':
    app.run(debug=True)
