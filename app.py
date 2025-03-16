from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

# Lista de cultivos y sus modelos
cultivos = {
    'Aguacate': ('modelos/modelo_clorofila_a_aguacate.h5', 'modelos/modelo_clorofila_b_aguacate.h5'),
    'Maíz': ('modelos/modelo_clorofila_a_maiz.h5', 'modelos/modelo_clorofila_b_maiz.h5'),
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

 
rangos = {
        "Café": {"A": [(7.56, 14.38, "Buena"), (2.27, 7.56, "Regular"), (0.09, 2.27, "Mala")],
                  "B": [(2.57, 6.03, "Buena"), (1.81, 2.57, "Regular"), (0.04, 1.81, "Mala")]},
        "Yuca": {"A": [(14.77, 21.36, "Buena"), (6.28, 14.77, "Regular"), (1.50, 6.28, "Mala")],
                  "B": [(5.39, 8.87, "Buena"), (2.03, 5.39, "Regular"), (0.87, 2.03, "Mala")]},
        "Aguacate": {"A": [(17.53, 21.62, "Buena"), (5.53, 17.53, "Regular"), (0.78, 5.53, "Mala")],
                      "B": [(7.61, 10.15, "Buena"), (1.67, 7.61, "Regular"), (0.44, 1.67, "Mala")]},
        "Plátano": {"A": [(0.33, 8.92, "Mala"), (8.93, 14.26, "Regular"), (14.27, 17.93, "Buena")],
                "B": [(0.17, 3.31, "Mala"), (3.32, 7.08, "Regular"), (7.09, 7.30, "Buena")]},
        "Caña": {"A": [(0.45, 6.36, "Mala"), (6.37, 10.51, "Regular"), (10.52, 15.08, "Buena")],
               "B": [(0.71, 2.22, "Mala"), (2.23, 3.96, "Regular"), (3.97, 4.02, "Buena")]},
        "Tomate": {"A": [(0.930, 5.42, "Mala"), (5.423, 7.8346, "Regular"), (7.835, 10.5335, "Buena")],
               "B": [(0.4281, 1.9998, "Mala"), (2.000, 4.4399, "Regular"), (4.440, 4.596, "Buena")]},
        "Maíz": {"A": [(-0.1738, 0.8808, "Mala"), (0.881, 9.9353, "Regular"), (9.936, 15.0635, "Buena")],
             "B": [(-0.2064, 0.9337, "Mala"), (0.934, 2.7077, "Regular"), (2.708, 4.7560, "Buena")]},
        "Papa": {"A": [(1.6638, 5.1699, "Mala"), (5.1700, 7.6260, "Regular"), (7.6261, 8.2543, "Buena")],
             "B": [(0.6297, 1.9529, "Mala"), (1.9530, 3.2974, "Regular"), (3.2975, 2.8359, "Buena")]}
    }

def clasificar_clorofila(cultivo, clorofila_a, clorofila_b):
    if cultivo not in rangos or clorofila_a is None or clorofila_b is None:
        return "Desconocido", "Desconocido"

    categoria_a = next((r[2] for r in rangos[cultivo]["A"] if r[0] <= clorofila_a <= r[1]), "Fuera de rango")
    categoria_b = next((r[2] for r in rangos[cultivo]["B"] if r[0] <= clorofila_b <= r[1]), "Fuera de rango")
    
    return categoria_a, categoria_b

def generar_grafico_clorofila(clorofila_a, clorofila_b, cultivo):
    if cultivo not in rangos:
        return None  

    fig, ax = plt.subplots(2, 1, figsize=(4, 2))  # Reduce el tamaño


    # Obtener el rango máximo de cada tipo de clorofila
    max_a = rangos[cultivo]["A"][-1][1]  # ✅ Último valor del rango
    max_b = rangos[cultivo]["B"][-1][1]  # ✅ Último valor del rango


    valores = [(clorofila_a, max_a, "Clorofila A"), (clorofila_b, max_b, "Clorofila B")]

    for i, (clorofila, max_valor, titulo) in enumerate(valores):
        # Crear degradado horizontal
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax[i].imshow(gradient, extent=[0, max_valor, 0, 1], cmap="RdYlGn", aspect="auto")

        # Flecha indicando el valor
        ax[i].annotate("",
               xy=(clorofila, 0.5),
               xytext=(clorofila, 1.5),
               arrowprops=dict(facecolor='black', arrowstyle="->", lw=2))


        # Configuración de etiquetas
        ax[i].set_xticks([0, max_valor])
        ax[i].set_xticklabels(["Deficiente", "Bueno"])
        ax[i].set_yticks([])  
        ax[i].set_title(titulo, fontsize=12, weight="bold")

    plt.tight_layout()
    
    # Guardar imagen en base64
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@app.route('/', methods=['GET', 'POST'])
def index():
    clorofila_a_predicha, clorofila_b_predicha = None, None
    categoria_a, categoria_b = None, None
    grafico_clorofila = None
    gari=None
    mnli=None
    gndvi=None
    cultivo_seleccionado=None

    if request.method == 'POST':
        try:
            cultivo_seleccionado = request.form.get('cultivo')
            gari = float(request.form.get('gari'))
            mnli = float(request.form.get('mnli'))
            gndvi = float(request.form.get('gndvi'))

            if cultivo_seleccionado not in modelos_cargados:
                return render_template('index.html', error="Cultivo no válido", cultivos=cultivos.keys())

            model_a, model_b = modelos_cargados[cultivo_seleccionado]
            nueva_muestra = pd.DataFrame({'GARI': [gari], 'MNLI': [mnli], 'GNDVI': [gndvi]})
            nueva_muestra_normalizada = scaler.transform(nueva_muestra)

            clorofila_a_predicha = float(model_a.predict(nueva_muestra_normalizada)[0][0])
            clorofila_b_predicha = float(model_b.predict(nueva_muestra_normalizada)[0][0])
           
            categoria_a, categoria_b = clasificar_clorofila(cultivo_seleccionado, clorofila_a_predicha, clorofila_b_predicha)

            # Generar gráfico de clorofila
            grafico_clorofila = generar_grafico_clorofila(clorofila_a_predicha, clorofila_b_predicha, cultivo_seleccionado)

        except Exception as e:
            return render_template('index.html', error=f"Error en la predicción: {str(e)}", cultivos=cultivos.keys())

    return render_template('index.html', 
                           clorofila_a=clorofila_a_predicha, 
                           clorofila_b=clorofila_b_predicha, 
                           categoria_a=categoria_a,
                           categoria_b=categoria_b,
                           grafico_clorofila=grafico_clorofila,
                           cultivos=cultivos.keys(),
                           gari=gari, mnli=mnli, gndvi=gndvi,
                           cultivo_seleccionado = cultivo_seleccionado)

if __name__ == '__main__':
    app.run(debug=True)